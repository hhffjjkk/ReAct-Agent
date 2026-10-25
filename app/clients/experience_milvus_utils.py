"""
诊断经验（diagnosis_experience）Milvus 集合封装。

字段（对齐任务文档）：
  exp_id (VARCHAR, PK) / query_text (VARCHAR) / query_embedding (FLOAT_VECTOR, dim=1024)
  tool_sequence (VARCHAR) / final_answer (VARCHAR) / success (BOOL)
  source_dept (VARCHAR) / create_time (INT64)

索引：query_embedding -> IVF_FLAT / COSINE，方向与 kb_chunks 的 COSINE 度量一致。
说明：复用现有 MilvusClient 单例，不新增基础设施。
"""
from typing import Any, Dict, List

from app.clients.milvus_utils import get_milvus_client
from app.conf.experience_config import experience_config
from app.core.logger import logger

_COLLECTION = experience_config.experience_memory_collection
_VECTOR_DIM = 1024


def _ensure_collection(client) -> bool:
    """集合不存在时按 schema 创建；返回是否可用"""
    if client is None:
        return False
    try:
        if client.has_collection(collection_name=_COLLECTION):
            return True

        from pymilvus import DataType

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="exp_id", datatype=DataType.VARCHAR,
                         max_length=512, is_primary=True)
        schema.add_field(field_name="query_text", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="query_embedding", datatype=DataType.FLOAT_VECTOR, dim=_VECTOR_DIM)
        schema.add_field(field_name="tool_sequence", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="final_answer", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="success", datatype=DataType.BOOL)
        schema.add_field(field_name="source_dept", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="create_time", datatype=DataType.INT64)

        index_params = client.prepare_index_params()
        # success 过滤很频繁，加标量倒排索引
        index_params.add_index(field_name="success", index_type="INVERTED")
        index_params.add_index(
            field_name="query_embedding",
            index_name="query_embedding_index",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )
        client.create_collection(collection_name=_COLLECTION, schema=schema, index_params=index_params)
        logger.info(f"[ExperienceMilvus] 已创建集合 {_COLLECTION}（IVF_FLAT/COSINE, dim={_VECTOR_DIM}）")
        return True
    except Exception as e:
        logger.error(f"[ExperienceMilvus] 确保集合存在失败：{e}")
        return False


def insert_experience(experience: Dict[str, Any]) -> bool:
    """写入一条经验；失败返回 False，不抛异常"""
    try:
        client = get_milvus_client()
        if not _ensure_collection(client):
            return False
        if hasattr(client, "upsert"):
            client.upsert(collection_name=_COLLECTION, data=[experience])
        else:
            client.insert(collection_name=_COLLECTION, data=[experience])
        client.flush(collection_name=_COLLECTION)
        logger.info(f"[ExperienceMilvus] 已写入经验：exp_id={experience.get('exp_id')}, "
                    f"success={experience.get('success')}")
        return True
    except Exception as e:
        logger.error(f"[ExperienceMilvus] 写入经验失败：{e}")
        return False


def search_experiences(query_embedding: List[float], topk: int,
                       threshold: float) -> List[Dict[str, Any]]:
    """
    检索成功经验（success == true），按 COSINE 相似度降序，过滤低于阈值的结果。
    集合不存在/为空/异常时返回空列表。
    """
    try:
        client = get_milvus_client()
        if not _ensure_collection(client):
            return []
        if client.has_collection(collection_name=_COLLECTION):
            try:
                client.load_collection(collection_name=_COLLECTION)
            except Exception:
                pass

        results = client.search(
            collection_name=_COLLECTION,
            data=[query_embedding],
            anns_field="query_embedding",
            search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=max(1, int(topk)),
            filter="success == true",
            output_fields=[
                "exp_id", "query_text", "tool_sequence",
                "final_answer", "success", "source_dept", "create_time",
            ],
        )
        hits = results[0] if results else []
        items: List[Dict[str, Any]] = []
        for hit in hits:
            distance = float(hit.get("distance") or 0.0)
            if distance < threshold:
                continue
            entity = hit.get("entity") or {}
            entity["score"] = distance
            # 防御性过滤：即使底层过滤失效，也不注入失败经验
            if not entity.get("success"):
                continue
            items.append(entity)
        return items
    except Exception as e:
        logger.warning(f"[ExperienceMilvus] 检索经验失败（已忽略）：{e}")
        return []

from __future__ import annotations

import json

from agentflow.config import QUERY_EXPANSION_TERMS_PATH


DEFAULT_DOMAIN_TERMS: dict[str, list[str]] = {
    "冷冻电镜": ["cryo-EM", "cryo electron microscopy"],
    "密度图": ["density map"],
    "局部框": ["local box", "local boxes"],
    "局部区域": ["local region", "local regions"],
    "残基": ["residue", "residues"],
    "相互作用": ["interaction", "interactions"],
    "空间依赖": ["spatial dependency", "spatial dependencies"],
    "分子动力学": ["molecular dynamics", "MD"],
    "动力学": ["dynamics"],
    "柔性": ["flexibility"],
    "结构柔性": ["structural flexibility"],
    "均方根波动": ["RMSF", "root mean square fluctuation"],
    "交叉验证": ["cross-validation"],
    "五折": ["5-fold", "five-fold"],
    "分辨率": ["resolution"],
    "低通滤波": ["low-pass filter"],
    "截止值": ["cutoff value", "cutoff values"],
    "相关性系数": ["correlation coefficient"],
    "Pearson": ["Pearson"],
    "零假设": ["null hypothesis"],
    "构象": ["conformation", "conformational"],
    "构象重排": ["conformational rearrangement"],
    "融合前": ["prefusion"],
    "融合后": ["post-fusion", "postfusion"],
    "中心螺旋": ["central helix"],
    "图谱": ["map", "maps"],
    "预测阶段": ["prediction stage"],
    "三维卷积神经网络": ["3D CNN", "three-dimensional CNN"],
    "蛋白质结合位点": ["protein binding site"],
}


def load_domain_terms() -> dict[str, list[str]]:
    if QUERY_EXPANSION_TERMS_PATH.exists():
        return json.loads(QUERY_EXPANSION_TERMS_PATH.read_text(encoding="utf-8"))
    return DEFAULT_DOMAIN_TERMS


def expand_query(query: str) -> str:
    additions: list[str] = []
    for zh_term, en_terms in load_domain_terms().items():
        if zh_term in query:
            additions.extend(en_terms)
    unique_additions = list(dict.fromkeys(additions))
    if not unique_additions:
        return query
    return f"{query} {' '.join(unique_additions)}"


def make_query_variants(query: str) -> dict[str, str]:
    expanded = expand_query(query)
    return {
        "original": query,
        "expanded": expanded,
    }

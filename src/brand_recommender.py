import argparse
import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRanker
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent / "data" / "sample"
PROJECT = Path(__file__).resolve().parent
ARTIFACTS = PROJECT / "artifacts"
NUMERIC_USER_COLUMNS = [
    "immediate_available_capital_10k_krw", "max_self_investment_10k_krw",
    "operating_reserve_10k_krw", "self_savings_amount_10k_krw",
    "family_funding_amount_10k_krw", "investor_funding_amount_10k_krw",
    "desired_loan_amount_10k_krw", "expected_interest_rate_pct", "loan_term_months",
    "current_debt_total_10k_krw", "current_monthly_debt_payment_10k_krw",
    "minimum_monthly_living_cost_10k_krw",
    "post_startup_household_monthly_after_tax_income_10k_krw",
    "current_franchise_store_count", "existing_store_monthly_net_profit_10k_krw",
    "target_monthly_after_tax_income_10k_krw", "target_payback_years",
    "actual_startup_timing_months",
]
CAT_COLUMNS = ["preferred_industry_1", "preferred_industry_2", "preferred_industry_3", "loan_status",
               "brand_name", "brand_industry", "brand_cat"]
COST_COLUMN = "총계"
COST_CLASS_COLUMN = "총계_s"


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def budget(profile):
    total = max(number(profile.get("immediate_available_capital_10k_krw")),
                number(profile.get("max_self_investment_10k_krw")))
    total += number(profile.get("family_funding_amount_10k_krw")) if profile.get("funding_family_flag") == "1" else 0
    total += number(profile.get("investor_funding_amount_10k_krw")) if profile.get("funding_investor_flag") == "1" else 0
    if profile.get("funding_loan_flag") == "1":
        total += number(profile.get("desired_loan_amount_10k_krw")) * (
            1 if profile.get("loan_status") in {"승인", "실행", "승인 완료"} else 0.5
        )
    return max(total - number(profile.get("operating_reserve_10k_krw")), 0)


def cat_distance(a, b):
    x, y = str(a).zfill(3)[-3:], str(b).zfill(3)[-3:]
    return sum(abs(int(i) - int(j)) for i, j in zip(x, y))


def normalize_industry(value):
    return " ".join(str(value).split())


def matched_preference(profile, brand):
    industry = normalize_industry(brand["industry_middle"])
    for i in range(1, 4):
        raw = profile.get(f"preferred_industry_{i}", "")
        preference = normalize_industry(raw)
        if not preference:
            continue
        if industry == preference:
            return raw
    return ""


def affordable_brand(profile, brand):
    cost = number(brand[COST_COLUMN]) / 10
    return bool(cost) and cost <= budget(profile)


def pair_features(profile, brand):
    row = {column: number(profile.get(column)) for column in NUMERIC_USER_COLUMNS}
    for column in CAT_COLUMNS[:4]:
        row[column] = str(profile.get(column, ""))
    row.update({
        "brand_name": brand["brand_name"],
        "brand_industry": brand["industry_middle"],
        "brand_cat": str(brand["cat"]).zfill(3),
        "brand_sales": number(brand["average_sales_per_3_3sqm"]),
        "brand_store_count": number(brand["store_count"]),
        "brand_cost_10k": number(brand[COST_COLUMN]) / 10,
        "brand_sales_class": number(brand["average_sales_per_3_3sqm_s"]),
        "brand_store_class": number(brand["store_count_s"]),
        "brand_cost_class": number(brand[COST_CLASS_COLUMN]),
    })
    preferred = [profile.get(f"preferred_industry_{i}", "") for i in range(1, 4)]
    row["preferred_rank"] = 3 - preferred.index(brand["industry_middle"]) if brand["industry_middle"] in preferred else 0
    row["available_budget_10k"] = budget(profile)
    row["affordable"] = int(bool(row["brand_cost_10k"]) and row["brand_cost_10k"] <= row["available_budget_10k"])
    row["budget_cost_ratio"] = row["available_budget_10k"] / max(row["brand_cost_10k"], 1)
    row["debt_to_budget"] = row["current_debt_total_10k_krw"] / max(row["available_budget_10k"], 1)
    return row


def make_pairs(profile_rows, brands, candidates):
    rows, labels, groups = [], [], []
    for group, (profile, positive, candidate_names) in enumerate(candidates(profile_rows, brands)):
        for name in candidate_names:
            rows.append(pair_features(profile, brands[name]))
            labels.append(int(name == positive))
            groups.append(group)
    return pd.DataFrame(rows), np.array(labels), np.array(groups)


def training_candidates(profiles, brands, matches, negatives=30, seed=42):
    rng = np.random.default_rng(seed)
    all_names = np.array(list(brands))
    for profile in profiles:
        positive = matches[profile["profile_id"]]
        target = brands[positive]
        preferred = {profile.get(f"preferred_industry_{i}", "") for i in range(1, 4)}
        hard = [name for name, brand in brands.items() if name != positive and (
            brand["industry_middle"] == target["industry_middle"]
            or brand["industry_middle"] in preferred
            or cat_distance(brand["cat"], target["cat"]) <= 1
        )]
        rng.shuffle(hard)
        chosen = hard[: negatives * 2 // 3]
        remaining = [name for name in all_names if name != positive and name not in chosen]
        chosen += list(rng.choice(remaining, size=negatives - len(chosen), replace=False))
        yield profile, positive, [positive] + chosen


def similarity_score(profile, first, candidate):
    preferred = {profile.get(f"preferred_industry_{i}", "") for i in range(1, 4)}
    score = 55 * (candidate["industry_middle"] == first["industry_middle"])
    score += 35 * (candidate["industry_middle"] in preferred)
    score += 30 - 10 * cat_distance(candidate["cat"], first["cat"])
    cost, funds = number(candidate[COST_COLUMN]) / 10, budget(profile)
    score += 20 if cost and cost <= funds else -20
    for weight, column in [(8, "average_sales_per_3_3sqm"), (4, "store_count"), (5, COST_COLUMN)]:
        score -= weight * abs(math.log1p(number(candidate[column])) - math.log1p(number(first[column])))
    return score


def additional_brands(profile, first_name, brands, candidates, count=3):
    first = brands[first_name]
    pool = [brands[name] for name in candidates if name != first_name]
    return sorted(pool, key=lambda brand: (-similarity_score(profile, first, brand), brand["brand_name"]))[:count]


def explain(profile, first, brand, rank):
    preference = matched_preference(profile, brand)
    cost_note = (
        "예상 창업비가 가용 예산 범위이며"
        if affordable_brand(profile, brand)
        else "예산 내 대안이 부족해 예상 창업비는 예산을 초과하지만"
    )
    if preference:
        return f"선호 업종 {preference}에 해당하고 {cost_note} 모델 적합도·매출·점포 규모를 반영"
    if rank == 1:
        return "브랜드 데이터에 직접 대응하는 선호 업종이 없어 모델 적합도로 선정"
    return f"선호 업종 후보가 부족해 1순위와 같은 {brand['industry_middle']} 업종에서 선정"


def load_data():
    users = read_csv(ROOT / "user_profiles.csv")
    brand_rows = read_csv(ROOT / "브랜드데이터.csv")
    matches = read_csv(ROOT / "브랜드_매칭결과.csv")
    brand_rows = brand_rows.drop_duplicates("brand_name", keep="last")
    brands = {row["brand_name"]: row.to_dict() for _, row in brand_rows.iterrows()}
    match_map = dict(zip(matches["profile_id"], matches["brand_name"]))
    return users, brands, match_map


def score_catalog(model, profile, brands):
    names = list(brands)
    features = pd.DataFrame([pair_features(profile, brands[name]) for name in names])
    return names, model.predict(features)


def select_brands(model, profile, brands):
    names, scores = score_catalog(model, profile, brands)
    score_map = dict(zip(names, scores))
    preferred_names = [name for name in names if matched_preference(profile, brands[name])]
    if preferred_names:
        affordable_names = [name for name in preferred_names if affordable_brand(profile, brands[name])]
        first_name = max(affordable_names or preferred_names, key=score_map.get)
        candidate_names = preferred_names
    else:
        first_name = names[int(np.argmax(scores))]
        first_industry = normalize_industry(brands[first_name]["industry_middle"])
        candidate_names = [name for name in names if normalize_industry(brands[name]["industry_middle"]) == first_industry]
    affordable_candidates = [
        name for name in candidate_names
        if name != first_name and affordable_brand(profile, brands[name])
    ]
    additional = additional_brands(profile, first_name, brands, affordable_candidates)
    if len(additional) < 3:
        used = {first_name, *(brand["brand_name"] for brand in additional)}
        remaining = [name for name in candidate_names if name not in used]
        additional += additional_brands(profile, first_name, brands, remaining, 3 - len(additional))
    return [brands[first_name]] + additional


def recommendation_rows(profile, selected):
    rows = []
    for rank, brand in enumerate(selected, 1):
            preference = matched_preference(profile, brand)
            rows.append({
                "profile_id": profile.get("profile_id", ""),
                "recommendation_rank": rank,
                "brand_name": brand["brand_name"],
                "industry_middle": brand["industry_middle"],
                "cat": str(brand["cat"]).zfill(3),
                "총계": brand[COST_COLUMN],
                "startup_cost_10k_krw": number(brand[COST_COLUMN]) / 10,
                "available_budget_10k_krw": budget(profile),
                "affordable": int(affordable_brand(profile, brand)),
                "matched_preference": preference,
                "selection_basis": "선호 업종" if preference else "동일 업종 대체",
                "reason": explain(profile, selected[0], brand, rank),
            })
    return pd.DataFrame(rows)


def recommend_profiles(model, profiles, brands):
    rows = []
    for profile in profiles:
        rows.extend(recommendation_rows(profile, select_brands(model, profile, brands)).to_dict("records"))
    return pd.DataFrame(rows)


def train(args):
    users, brands, matches = load_data()
    train_ids, test_ids = train_test_split(
        users["profile_id"], test_size=args.test_size, random_state=args.seed,
        stratify=users["profile_id"].map(matches),
    )
    train_profiles = users.set_index("profile_id").loc[train_ids].reset_index().to_dict("records")
    test_profiles = users.set_index("profile_id").loc[test_ids].reset_index().to_dict("records")
    candidates = lambda profiles, brand_map: training_candidates(profiles, brand_map, matches, args.negatives, args.seed)
    x_train, y_train, groups = make_pairs(train_profiles, brands, candidates)
    model = CatBoostRanker(
        loss_function="YetiRankPairwise", iterations=args.iterations, depth=8,
        learning_rate=0.08, random_seed=args.seed, verbose=50, allow_writing_files=False,
    )
    print(f"training_pairs={len(x_train):,} test_profiles={len(test_profiles):,}", flush=True)
    model.fit(x_train, y_train, group_id=groups, cat_features=CAT_COLUMNS)

    hits1 = hits4 = industry_hits = 0
    reciprocal_ranks, output_rows = [], []
    for index, profile in enumerate(test_profiles, 1):
        names, scores = score_catalog(model, profile, brands)
        order = np.argsort(-scores)
        truth = matches[profile["profile_id"]]
        truth_rank = int(np.where(np.array(names)[order] == truth)[0][0]) + 1
        selected = select_brands(model, profile, brands)
        first = selected[0]["brand_name"]
        final = [brand["brand_name"] for brand in selected]
        hits1 += first == truth
        hits4 += truth in final
        industry_hits += brands[first]["industry_middle"] == brands[truth]["industry_middle"]
        reciprocal_ranks.append(1 / truth_rank)
        output_rows.extend(recommendation_rows(profile, selected).to_dict("records"))
        if index % 250 == 0:
            print(f"evaluated={index:,}/{len(test_profiles):,}", flush=True)

    ARTIFACTS.mkdir(exist_ok=True)
    model.save_model(ARTIFACTS / "brand_ranker.cbm")
    joblib.dump({"brands": brands, "feature_columns": list(x_train.columns)}, ARTIFACTS / "metadata.joblib")
    metrics = {
        "test_profiles": len(test_profiles), "catalog_brands": len(brands),
        "labeled_brands": len(set(matches.values())), "hit_at_1": hits1 / len(test_profiles),
        "system_hit_at_4": hits4 / len(test_profiles),
        "industry_hit_at_1": industry_hits / len(test_profiles),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
    }
    (PROJECT / "evaluation.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(output_rows).to_csv(PROJECT / "test_recommendations.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def recommend(args):
    model = CatBoostRanker()
    model.load_model(ARTIFACTS / "brand_ranker.cbm")
    metadata = joblib.load(ARTIFACTS / "metadata.joblib")
    profiles = read_csv(Path(args.input)).to_dict("records")
    result = recommend_profiles(model, profiles, metadata["brands"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved={args.output} rows={len(result):,}")


def evaluate_existing(args):
    users, brands, matches = load_data()
    model = CatBoostRanker()
    model.load_model(ARTIFACTS / "brand_ranker.cbm")
    test_ids = read_csv(Path(args.test_ids_from))["profile_id"].drop_duplicates()
    profiles = users.set_index("profile_id").loc[test_ids].reset_index().to_dict("records")
    rows, hit1, hit4, industry_hit, preferred_top1 = [], 0, 0, 0, 0
    for index, profile in enumerate(profiles, 1):
        selected = select_brands(model, profile, brands)
        names = [brand["brand_name"] for brand in selected]
        truth = matches[profile["profile_id"]]
        hit1 += names[0] == truth
        hit4 += truth in names
        industry_hit += normalize_industry(selected[0]["industry_middle"]) == normalize_industry(brands[truth]["industry_middle"])
        preferred_top1 += bool(matched_preference(profile, selected[0]))
        rows.extend(recommendation_rows(profile, selected).to_dict("records"))
        if index % 250 == 0:
            print(f"evaluated={index:,}/{len(profiles):,}", flush=True)
    result = pd.DataFrame(rows)
    counts = result.groupby("profile_id").size()
    top_rows = result[result["recommendation_rank"] == 1]
    additional = result[result["recommendation_rank"] > 1]
    metrics = {
        "test_profiles": len(profiles),
        "hit_at_1": hit1 / len(profiles),
        "system_hit_at_4": hit4 / len(profiles),
        "industry_hit_at_1": industry_hit / len(profiles),
        "top1_preference_aligned": preferred_top1 / len(profiles),
        "additional_preference_aligned": float((additional["matched_preference"] != "").mean()) if len(additional) else 0,
        "profiles_with_4_recommendations": float((counts == 4).mean()),
        "average_recommendations_per_profile": float(counts.mean()),
        "top1_affordable_rate": float(top_rows["affordable"].mean()),
        "all_recommendations_affordable_rate": float(result["affordable"].mean()),
    }
    result.to_csv(PROJECT / "test_recommendations_preference_fixed.csv", index=False, encoding="utf-8-sig")
    (PROJECT / "evaluation_preference_fixed.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--iterations", type=int, default=300)
    train_parser.add_argument("--negatives", type=int, default=30)
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(run=train)
    recommend_parser = commands.add_parser("recommend")
    recommend_parser.add_argument("--input", required=True)
    recommend_parser.add_argument("--output", default=str(PROJECT / "recommendations.csv"))
    recommend_parser.set_defaults(run=recommend)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--test-ids-from", default=str(PROJECT / "test_recommendations.csv"))
    evaluate_parser.set_defaults(run=evaluate_existing)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd

def load_pairs(csv_path: Path) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Returns:
      - by_id: dict id_pessoa -> list of image paths
      - id2name: dict id_pessoa -> person name
    Expects CSV columns: id_pessoa,name,img
    """
    df = pd.read_csv(csv_path)
    by_id = defaultdict(list)
    id2name = {}
    for _, row in df.iterrows():
        img = str(row["img"])
        idp = str(row["id_pessoa"])
        name = str(row["name"]) if "name" in row and not pd.isna(row["name"]) else ""
        by_id[idp].append(img)
        if idp not in id2name:
            id2name[idp] = name
    return dict(by_id), id2name

def all_anchor_positive_pairs(by_id: Dict[str, List[str]]) -> List[Tuple[str,str,str]]:
    """
    returns list of (anchor_img, pos_img, anchor_id)
    only for ids with >=2 images
    """
    pairs = []
    for idp, imgs in by_id.items():
        if len(imgs) < 2:
            continue
        for a in imgs:
            for p in imgs:
                if a == p:
                    continue
                pairs.append((a, p, idp))
    return pairs

def build_reverse_map(by_id: Dict[str,List[str]]) -> Dict[str,str]:
    rev = {}
    for idp, imgs in by_id.items():
        for img in imgs:
            rev[img] = idp
    return rev

def build_triplets_improved(
    csv_path: Path,
    out_dir: Path,
    seed: int = 5932,
    fraction_single_neg: float = 0.5,
    max_triplets: int = None
) -> Tuple[int,int]:
    """
    Improved version with precomputed reverse map and proper negative sampling per anchor id.
    Outputs CSV with columns:
      anchor, positive, negative, anchor_id, pos_id, neg_id, anchor_name, pos_name, neg_name
    """
    random.seed(seed)
    by_id, id2name = load_pairs(csv_path)
    rev = build_reverse_map(by_id)

    single_ids = [i for i, imgs in by_id.items() if len(imgs) == 1]
    single_imgs = [by_id[i][0] for i in single_ids]
    all_imgs = [img for imgs in by_id.values() for img in imgs]

    ap_pairs = all_anchor_positive_pairs(by_id)
    random.shuffle(ap_pairs)

    triplets_set: Set[Tuple[str,str,str]] = set()
    triplets_list = []

    total_possible = len(ap_pairs) * (len(all_imgs) - 2) if len(all_imgs) > 2 else len(ap_pairs)
    target = max_triplets or total_possible

    for a_img, p_img, aid in ap_pairs:
        if len(triplets_list) >= target:
            break

        # decide negative source (singleton vs any)
        use_single = random.random() < fraction_single_neg and len(single_imgs) > 0

        neg_img = None
        attempts = 0
        while attempts < 200:
            attempts += 1
            if use_single:
                candidate = random.choice(single_imgs)
            else:
                candidate = random.choice(all_imgs)
            if candidate == a_img or candidate == p_img:
                continue
            # candidate must belong to different id
            if rev.get(candidate) == aid:
                continue
            neg_img = candidate
            break

        if neg_img is None:
            continue

        trip = (a_img, p_img, neg_img)
        if trip in triplets_set:
            continue
        triplets_set.add(trip)

        neg_id = rev.get(neg_img, "")
        anchor_name = id2name.get(aid, "")
        pos_name = id2name.get(aid, "")
        neg_name = id2name.get(neg_id, "")

        triplets_list.append((
            a_img, p_img, neg_img,
            aid, aid, neg_id,
            anchor_name, pos_name, neg_name
        ))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (csv_path.stem + "_triplets.csv")
    df_out = pd.DataFrame(triplets_list, columns=[
        "anchor","positive","negative",
        "anchor_id","pos_id","neg_id",
        "anchor_name","pos_name","neg_name"
    ])
    df_out.to_csv(out_path, index=False)

    return len(triplets_list), len(triplets_set)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV (id_pessoa,name,img)")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory for triplet CSV")
    parser.add_argument("--seed", type=int, default=5932)
    parser.add_argument("--fraction-single-neg", type=float, default=0.5, help="Fraction of negatives sampled from singleton IDs")
    parser.add_argument("--max-triplets", type=int, default=None, help="Max number of triplets to generate")
    args = parser.parse_args()

    generated, unique = build_triplets_improved(
        args.csv, args.out_dir, seed=args.seed,
        fraction_single_neg=args.fraction_single_neg,
        max_triplets=args.max_triplets
    )
    print(f"Generated triplets: {generated} (unique: {unique}). Saved to {args.out_dir / (args.csv.stem + '_triplets.csv')}")

if __name__ == "__main__":
    main()
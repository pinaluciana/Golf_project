from pathlib import Path
import pandas as pd


def main():
    folder = Path(__file__).parent
    files = sorted(folder.glob("*.csv"))

    dfs = []

    for f in files:
        if "combined" in f.name:
            continue

        year = f.name.split("_")[0]

        df = pd.read_csv(f)

        #remove players who didn't make the cut, got disqualified("DQ") or withdrew("WD")
        df = df[~df["position"].isin(["CUT", "DQ", "WD"])]
        
        df["year"] = int(year)
        dfs.append(df)

    all_us_open = pd.concat(dfs, ignore_index=True)

    project_root = folder.parents[2]

    processed_folder = project_root / "data" / "processed" / "US_Open"
    processed_folder.mkdir(parents=True, exist_ok=True)

    output_path = processed_folder / "US_Open_combined_data.csv"
    all_us_open.to_csv(output_path, index=False)

    print(f"Saved combined file to {output_path}")
    print(all_us_open.head())


if __name__ == "__main__":
    main()

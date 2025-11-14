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
        df["year"] = int(year)
        dfs.append(df)
        
    all_pga = pd.concat(dfs, ignore_index=True)
    
    project_root = folder.parents[2]
    
    processed_folder = project_root / "data/processed/PGA_Championship"
    processed_folder.mkdir(parents=True, exist_ok=True)
    
    output_path = processed_folder / "PGA_combined_data.csv"
    all_pga.to_csv(output_path, index=False)
    
    print(f"Saved combined file to {output_path}")
    print(all_pga.head())

if __name__ == "__main__":
    main()
# Trump GPT

## Setup

### Configure DVC remote credentials

```bash
dvc remote modify origin --local auth basic
dvc remote modify origin --local user YOUR_DAGSHUB_USERNAME
dvc remote modify origin --local password YOUR_DAGSHUB_TOKEN
```

Get your token from: DagsHub → Settings → Access Tokens

### Pull the data

```bash
dvc pull
```

This downloads all DVC-tracked files into `data/raw/`.

---

## Project structure

```
├── .dvc/             # DVC configuration
├── data/
│   └── raw/          # raw data, tracked by DVC
├── models/           # trained models, tracked by DVC
├── src/              # training and processing scripts
├── dvc.yaml          # pipeline definition (added later)
└── README.md
```

---

## Daily workflow

### Make changes and reproduce the pipeline

```bash
# after modifying scripts or adding data
dvc repro
```

### Push changes

```bash
dvc push          # push data/models to DagsHub
git add .
git commit -m "your message"
git push
```

---

## Adding new data

1. Place the file in `data/raw/`
2. Track it with DVC:

```bash
dvc add data/raw/your_file.csv
```

3. Commit and push:

```bash
git add data/raw/your_file.csv.dvc data/raw/.gitignore
git commit -m "data: add your_file.csv"
git push origin main
dvc push
```

---

## Running the pipeline

> Pipeline stages will be defined in `dvc.yaml` as the project develops.

Once defined, run the full pipeline with:

```bash
dvc repro
```

DVC will only re-run stages whose dependencies have changed.

---

## Notes

- Never commit raw data files directly to Git — always use `dvc add`
- `.dvc/config.local` holds your credentials and is gitignored by default
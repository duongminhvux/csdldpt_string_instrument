# String Instrument Search

## Structure
- `data/dataset/`: WAV dataset grouped as needed
- `data/query/`: WAV files used for search input
- `database/schema.sql`: MySQL schema
- `src/`: core source code
- `scripts/`: entry scripts
- `results/`: output files

## Next steps
1. Put dataset WAV files into `data/dataset/`
2. Put query WAV files into `data/query/`
3. Create MySQL tables with `database/schema.sql`
4. Implement feature extraction in `src/feature_extraction.py`
5. Complete DB logic in `src/database_manager.py`
6. Complete retrieval logic in `src/retrieval.py`



## Cau hinh ket noi MySQL
Chi can sua 1 lan trong file `src/config.py` o bien `DB_CONFIG`.

## Ket qua query
Sau khi chay `python scripts/run_query.py`:
- Ket qua bang se duoc luu vao `results/top5_results.csv`
- 5 file wav giong nhat se duoc copy vao `results/top5_wav/<ten_file_query>/`

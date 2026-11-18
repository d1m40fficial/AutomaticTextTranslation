# Данные

Для бенчмарка используется параллельный корпус **english-russian dictionary for machine translation** с Kaggle:

> https://www.kaggle.com/datasets/hijest/englishrussian-dictionary-for-machine-translate/data

Датасет содержит **363 386** пар предложений в формате `TSV`:

```
English sentence<TAB>Русское предложение
```

Файл `rus.txt` (~60 МБ) **не хранится в репозитории** — он игнорируется git и скачивается отдельно.

## Как получить файл

Через Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download -d hijest/englishrussian-dictionary-for-machine-translate -p data --unzip
```

Или вручную: открыть ссылку, нажать **Download**, распаковать архив и положить `rus.txt` в эту папку.

Итоговая структура:

```
data/
└── rus.txt
```

## Запуск оценки

```bash
python evaluation/evaluate_bleu.py --data data/rus.txt --sample 10000
```

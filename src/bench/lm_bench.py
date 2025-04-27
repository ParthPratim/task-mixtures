import csv


class LM_Bench:
    """
    Design this class to conduct LM_BENCH evaluation effortlessly
    """

    def __init__(self):
        pass

    def extract_table(self, filename, outfilename):
        with open(filename, "r") as f:
            csv_lines = []
            for line in f:
                g = line.strip()[1:-1]
                cols = g.split("|")
                data_cols = []
                for col in cols:
                    clean_col = col.strip()
                    if len(clean_col) == 0:
                        continue
                    else:
                        data_cols.append(clean_col)

                csv_lines.append(data_cols)

            with open(outfilename, "w") as csvf:
                writer = csv.writer(csvf)
                writer.writerows(csv_lines)


o = LM_Bench()
o.extract_table("tmp/mmlu_uniform_1.txt", "artifacts/bench/mmlu_uniform_per_task.csv")
o.extract_table(
    "tmp/mmlu_uniform_2.txt", "artifacts/bench/mmlu_uniform_per_category.csv"
)

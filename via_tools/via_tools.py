import ast
import csv
import warnings


def via32dict(csv_name):
    D = dict()
    with open(csv_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for fields in reader:
            if len(fields) < 1 or fields[0].startswith("# CSV_HEADER = "):
                fields[0] = fields[0].split("# CSV_HEADER = ")[-1]
                break
        reader = csv.DictReader(csvfile, fieldnames=fields)
        # converts each new line into a dictionary based on the header field given previously
        for i, entry in enumerate(reader):
            for key in entry.keys():
                try:
                    entry[key] = ast.literal_eval(entry[key])
                except SyntaxError as e:
                    pass
            assert len(entry["file_list"]) > 0, "file_list must be non-empty"
            if len(entry["file_list"]) != 1:
                warnings.warn("file_list has multiple entries, using first one.", DeprecationWarning)
            file_name = entry.pop("file_list")[0]
            if file_name in D:
                attributes = D[file_name]
                attributes.append(entry)
            else:
                attributes = [entry]
            D.update({file_name: attributes})
    return D

def print_sudoku(quiz):
    out = ""
    for idx_row, row in enumerate(quiz):
        if idx_row == 3 or idx_row == 6:
            out += F"{'-' * 28}\n"
        for e_idx, d in enumerate(row):
            if e_idx == 3 or e_idx == 6:
                out += "|"
            if d:
                out += F" {d} "
            else:
                out += "   "
        out += "\n"
    print(out[:-1])

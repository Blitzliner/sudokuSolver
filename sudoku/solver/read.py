import pandas
from . import helper

class SudokuReader:
    _data = None
    _current_idx = 0
    _max_sudokus = None
    _current_sudoku = None

    def __init__(self, file="sudokus.csv", nrows=_max_sudokus):
        self._data = pandas.read_csv(file, dtype=str,)
        self._max_sudokus = self._data.shape[0]

    def reset_reader(self):
        self._current_idx = 0

    def next(self):
        if self._data is not None and self._current_idx < self._max_sudokus:
            quiz = self._data.iloc[self._current_idx, 1]
            #solution = self._data.iloc[self._current_idx, 2]
            self._current_idx += 1
            arr = [int(c) for c in quiz]
            self._current_sudoku = [arr[(r-1)*9:r*9] for r in range(1, 10)]
            return self._current_sudoku
        return None


if __name__ == '__main__':
    reader = SudokuReader()
    sudoku = reader.next()
    helper.print_sudoku(sudoku)

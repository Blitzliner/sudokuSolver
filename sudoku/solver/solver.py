from . import read
from . import helper

class SudokuSolver:
    _lt_box = [0, 0, 0, 3, 3, 3, 6, 6, 6]

    def solve(self, bo):
        find = self._find_empty(bo)
        if not find:
            return True
        else:
            row, col = find
        for i in range(1, 10):
            if self._valid(bo, i, (row, col)):
                bo[row][col] = i
                if self.solve(bo):
                    return True
                bo[row][col] = 0
        return False

    def _valid(self, bo, num, pos):
        # Check row
        for i in range(len(bo[0])):
            if bo[pos[0]][i] == num and pos[1] != i:
                return False
        # Check column
        for i in range(len(bo)):
            if bo[i][pos[1]] == num and pos[0] != i:
                return False
        # Check box
        box_x = self._lt_box[pos[1]]
        box_y = self._lt_box[pos[0]]
        for i in range(box_y, box_y + 3):
            for j in range(box_x, box_x + 3):
                if bo[i][j] == num and (i, j) != pos:
                    return False
        return True

    def _find_empty(self, bo):
        for i in range(len(bo)):
            for j in range(len(bo[0])):
                if bo[i][j] == 0:
                    return i, j  # row, col
        return None


def timing_wrapper():
    solver.solve(reader.next())


if __name__ == '__main__':
    reader = read.SudokuReader()
    solver = SudokuSolver()
    quiz = reader.next()
    helper.print_sudoku(quiz)
    solver.solve(quiz)
    print("Solved:")
    helper.print_sudoku(quiz)

    # do timings
    if True:
        import timeit
        max_iterations = 1000
        reader = read.SudokuReader()
        reader.reset_reader()
        time = timeit.timeit(timing_wrapper, number=max_iterations)
        print(F" time for {max_iterations} total {time:.4}s in avg {(time / max_iterations)*1000:.3f}ms")

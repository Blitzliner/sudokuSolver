from sudoku.solver.solver import SudokuSolver
from sudoku.solver.helper import print_sudoku
from sudoku import read_image  # sudoku.read_sudoku
import os


if __name__ == '__main__':
    # file = os.path.abspath('sudoku/data/image1005.original.jpg')  # seems to hard? takes over 10 seconds to solve
    file = os.path.abspath('sudoku/data/image1088.original.jpg')
    quiz = read_image.read_sudoku(file, debug=True)

    solver = SudokuSolver()
    print_sudoku(quiz)

    solver.solve(quiz)
    print("Solved:")
    print_sudoku(quiz)

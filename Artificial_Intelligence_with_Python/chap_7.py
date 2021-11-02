import sys


def heuristic_search(option=1):
    if option == 1:
        ''' Constructing a string using greedy search '''
        print("Example: I say Hello world")
        pass
    elif option == 2:
        ''' Solving a problem with constraints '''
        pass
    elif option == 3:
        ''' Solving the region-coloring problem '''
        pass
    elif option == 4:
        ''' Building an 8-puzzle solver '''
        pass
    elif option == 5:
        ''' Building a maze solver '''
        pass
    pass


def main():
    heuristic_search(int(sys.argv[1]))


if __name__ == "__main__":
    main()
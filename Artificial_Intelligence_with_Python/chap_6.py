import sys


def logic_programming(option=1):
    if option == 1:
        ''' Matching mathematical expressions '''
        print("Example: I say Hello world")
        pass
    elif option == 2:
        ''' Validating primes '''
        pass
    elif option == 3:
        ''' Parsing a family tree '''
        pass
    elif option == 4:
        ''' Analyzing geography '''
        pass
    elif option == 5:
        ''' Building a puzzle solver '''
        pass
    pass


def main():
    logic_programming(int(sys.argv[1]))


if __name__ == "__main__":
    main()
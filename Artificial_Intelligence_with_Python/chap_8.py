import sys


def genetic_algo(option=1):
    if option == 1:
        ''' Generating a bit pattern with predefined parameters '''
        print("Example: I say Hello world")
        pass
    elif option == 2:
        ''' Visualizing the evolution '''
        pass
    elif option == 3:
        ''' Solving the symbol regression problem '''
        pass
    elif option == 4:
        ''' Building an intelligent robot controller '''
        pass
    pass


def main():
    genetic_algo(int(sys.argv[1]))


if __name__ == "__main__":
    main()
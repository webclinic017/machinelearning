import sys


def figureout_sequential_data(option=1):
    if option == 1:
        ''' Handling time-series data with Pandas '''
        print("Example: I say Hello world")
        pass
    elif option == 2:
        ''' Slicing time-series data '''
        pass
    elif option == 3:
        ''' Operating on time-series data '''
        pass
    elif option == 4:
        ''' Extracting statistics from time-series data '''
        pass
    elif option == 5:
        ''' data using Hidden Markov Models '''
        pass
    elif option == 6:
        ''' Identifying alphabet sequences with Conditional Random Fields '''
        pass
    else:
        ''' Stock market analysis '''
        pass
    pass


def main():
    figureout_sequential_data(int(sys.argv[1]))


if __name__ == "__main__":
    main()
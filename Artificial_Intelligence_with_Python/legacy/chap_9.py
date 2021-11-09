import sys


def ai_games(option=1):
    if option == 1:
        ''' Building a bot to play Last Coin Standing '''
        print("Example: I say Hello world")
        pass
    elif option == 2:
        ''' Building a bot to play Tic-Tac-Toe '''
        pass
    elif option == 3:
        ''' Building two bots to play Connect Four against each other '''
        pass
    elif option == 4:
        ''' Building two bots to play Hexapawn against each other '''
        pass
    elif option == 5:
        ''' option_purpose '''
        pass
    else:
        ''' option_purpose '''
        pass
    pass


def main():
    ai_games(int(sys.argv[1]))


if __name__ == "__main__":
    main()
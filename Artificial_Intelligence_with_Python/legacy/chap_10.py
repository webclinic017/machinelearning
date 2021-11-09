import sys


def nlp(option=1):
    if option == 1:
        ''' Tokenizing text data '''
        print("Example: I say Hello world")
        pass
    elif option == 2:
        ''' Converting words to their base forms using stemming '''
        pass
    elif option == 3:
        ''' Converting words to their base forms using lemmatization '''
        pass
    elif option == 4:
        ''' Dividing text data into chunks '''
        pass
    elif option == 5:
        ''' Extracting the frequency of terms using a Bag of Words model '''
        pass
    elif option == 6:
        ''' Building a category predictor '''
        pass
    elif option == 7:
        ''' Constructing a gender identifier '''
        pass
    elif option == 8:
        ''' Building a sentiment analyzer '''
        pass
    else:
        ''' Topic modeling using Latent Dirichlet Allocation '''
        pass
    pass


def main():
    nlp(int(sys.argv[1]))


if __name__ == "__main__":
    main()
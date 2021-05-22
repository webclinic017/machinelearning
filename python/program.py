# Entering Expressions into the Interactive Shell
# Done: python .... quit()
# ---------------------------------------------------

# # The Integer, Floating-Point, and String Data Types
# print(2)
# a = 2.54545
# print(f'{a}')
# print('{}'.format(a))
# print('hello word')

# # ---------------------------------------------------
# # String Concatenation and Replication
# # Storing Values in Variables

# print(2*3)
# print(2**3)
# print('Hello '+'world')
# print('Hello '+str(87493))

# print('Alice '*5)
# print([1]*3 + [2]*6)

# # TypeError: unsupported operand type(s) for *: 'set' and 'int'
# # print([1]*3 + {2}*6)

# # TypeError: 'int' object is not iterable
# # print(tuple(2)*5)

# print(tuple([2])*5)
# tmp = {2: 3, 3: 4}

# # TypeError: unsupported operand type(s) for *: 'dict' and 'int'
# # print(tmp*3)

# # ---------------------------------------------------
# # This program says hello and asks for my name.
# # Dissecting Your Program

# print('Hello world!')
# print('What is your name?')  # ask for their name
# myName = input()
# print('It is good to meet you, ' + myName)
# print('The length of your name is:')
# print(len(myName))
# print('What is your age?')  # ask for their age
# myAge = input()
# print('You will be ' + str(int(myAge) + 1) + ' in a year.')

# ---------------------------------------------------
# Boolean Values: True False
# # Comparison Operators
# # Boolean Operators: and or not
# # Mixing Boolean and Comparison Operators: < and > ... Eg: (1, 5)
# # Elements of Flow Control: Conditions + Blocks
# # Flow Control: if else elif, while, break, continue
# while True:
#     print('Who are you?')
#     name = input()
#     if name != 'Hung':
#         continue
#     print('Hello, Hung. What is the password? (It is a fish.)')
#     password = input()
#     if password == 'swordfish':
#         break
# print('Access granted')

# truthy and falsey values
# ---------------------------------------------------
# for Loops and the range()

# print('My name is')
# for i in range(5):
#     print('Jimmy Five Times (' + str(i) + ')')
# ---------------------------------------------------
# An Equivalent while Loop
# print('My name is')
# i = 0
# while i < 5:
#     print('Jimmy Five Times (' + str(i) + ')')
#     i = i + 1
# # ---------------------------------------------------
# # Starting, Stopping, and Stepping Arguments to range()

# for i in range(12, 26, 3):
#     print(i)
# # ---------------------------------------------------
# # Importing Modules
# # sys.exit()
# import sys
# while True:
#     print('Type exit to exit.')
#     response = input()
#     if response == 'exit':
#         sys.exit()
#     print('You typed ' + response + '.')
# # ---------------------------------------------------
# # def Statements with Parameters

# def hello(name='Hung'):
#     print('hello '+name)
#     pass

# Keyword Arguments
# hello(name='Ngoc')
# # ---------------------------------------------------
# # Return Values and return Statements
# import random

# for _ in range(10):
#     print(random.randint(1, 9))
# # ---------------------------------------------------
# The None Value

# # ---------------------------------------------------
# # Keyword Arguments
# print('cats', 'dogs', 'mice')
# print('cats', 'dogs', 'mice', sep=',')
# # ---------------------------------------------------
# Local and Global Scope
# nonlocal and global
# # ---------------------------------------------------
# # Exception Handling
# try:
#     pass
# except Exception as e:
#     pass

# # ----------------------
# def spam(divideBy):
#     try:
#         return 42/divideBy
#     except ZeroDivisionError:
#         print('Error: Invalid argument.')


# print(spam(2))
# print(spam(12))
# print(spam(0))
# print(spam(1))

# # # ---------------------------------------------------
# # The List Data Type []

# # Getting Individual Values in a List with Indexes

# # Negative Indexes

# # # Getting Sublists with Slices (index range)
# # spam = ['cat', 'bat', 'rat', 'elephant']
# # print(spam[1:])
# # print(len(spam))

# # Changing Values in a List with Indexes

# # Concatenation and List Replication
# spam = ['cat', 'bat', 'rat', 'elephant']
# print(spam * 2)

# print([1, 2, 3] + ['A', 'B', 'C'])

# # Removing Values from Lists with del Statements
# del spam[3]
# print(spam)

# # # ---------------------------------------------------
# # Working with Lists
# catNames = []
# while True:
#     print('Enter the name of cat ' + str(len(catNames) + 1) +
#           ' (Or enter nothing to stop.):')
#     name = input()
#     if name == '':
#         break
#     catNames = catNames + [name]
# print('The cat names are:')
# for name in catNames:
#     print(' ' + name)

# # # # ---------------------------------------------------
# # Using for Loops with Lists
# spam = ['cat', 'bat', 'rat', 'elephant']
# for i in range(len(spam)):
#     print(spam[i])
# print()
# for item in spam:
#     print(item)
# # # # ---------------------------------------------------
# The in and not in Operators
# # # # ---------------------------------------------------
# # The Multiple Assignment Trick
# cat = ['fat', 'black', 'loud']
# size, color, disposition = cat
# print(size, color, disposition, sep=", ")

# # # # ---------------------------------------------------
# # Augmented Assignment Operators
# spam = 1
# while True:
#     print(spam)
#     if spam == 10:
#         break
#     spam += 1

# # # # ---------------------------------------------------
# # Methods: same thing as a function, except it is “called on” a value.
# spam = ['hello', 'hi', 'howdy', 'heyas']
# print(spam.index('hi'))
# spam.append('haha')
# print(spam)
# spam.sort()
# print(spam)
# try:
#     spam.remove('heyas')
# except Exception as e:
#     print(e)
#     pass
# # # # ---------------------------------------------------
# # Magic 8 Ball with a List
# import random

# messages = ['It is certain',
#             'It is decidedly so',
#             'Yes definitely',
#             'Reply hazy try again',
#             'Ask again later',
#             'Concentrate and ask again',
#             'My reply is no',
#             'Outlook not so good',
#             'Very doubtful']

# for _ in range(len(messages)):
#     print(messages[random.randint(0, len(messages)-1)])
# # # # ---------------------------------------------------
# List-like Types: Strings and Tuples

# Mutable and Immutable Data Types
# 'str' object does not support item assignment: so it Immutable
# slicing and concatenation to make a new str

# # # # ---------------------------------------------------
# tuple : Immutable ()

# Converting Types with the list() and tuple() Functions
# # # # ---------------------------------------------------
# References
# spam = [0, 1, 2, 3, 4, 5]
# tmp = spam
# tmp[-1] = 6
# print(spam)
# # tmp = tmp * 2
# # print(tmp)


# def eggs(someParams):
#     # copy of the reference used for the parameter
#     someParams.append('Hello')


# spam = [1, 2, 3]
# eggs(spam)
# print(spam)
# # # # ---------------------------------------------------
# The copy Module’s copy() and deepcopy() Functions
# # # # # ---------------------------------------------------
# grid = [['.', '.', '.', '.', '.', '.'],
#         ['.', 'O', 'O', '.', '.', '.'],
#         ['O', 'O', 'O', 'O', '.', '.'],
#         ['O', 'O', 'O', 'O', 'O', '.'],
#         ['.', 'O', 'O', 'O', 'O', 'O'],
#         ['O', 'O', 'O', 'O', 'O', '.'],
#         ['O', 'O', 'O', 'O', '.', '.'],
#         ['.', 'O', 'O', '.', '.', '.'],
#         ['.', '.', '.', '.', '.', '.']]

# output = []
# # # # ---------------------------------------------------
# # Dictionaries and Structuring Data
# # Dictionaries vs. Lists
# # keys(), values(), and items()
# # Checking Whether a Key or Value Exists
# spam = {'name': 'Zophie', 'age': 7}
# print('name' in spam.keys())

# # The get() Method: return 0 if 'name' does not exist in spam
# print(spam.get('name', 0))

# # setdefault() Method: set Hung at 'another name'
# # if 'another name' does not exist
# print(spam.setdefault('another name', 'Hung'))
# print(spam)
# # # # ---------------------------------------------------
# import pprint
# # characterCount
# message = 'It was a bright cold day in April, and\
#      the clocks were striking thirteen.'
# count = {}
# for char in message:
#     count.setdefault(char, 0)
#     count[char] += 1
# # print(count)
# pprint.pprint(count)

# # # # ---------------------------------------------------
# Using Data Structures to Model Real-World Things
# A Tic-Tac-Toe Board

# # # # ---------------------------------------------------
# # Nested Dictionaries and Lists
# allGuests = {'Alice': {'apples': 5, 'pretzels': 12},
#              'Bob': {'ham sandwiches': 3, 'apples': 2},
#              'Carol': {'cups': 3, 'apple pies': 1}}


# def totalBrought(guests, item):
#     num = 0
#     for k, v in allGuests.items():
#         # v is Nested Dictionaries
#         # print(v)
#         num += v.get(item, 0)
#     return num


# print('Number of things being brought:')
# print(' - Apples ' + str(totalBrought(allGuests, 'apples')))
# print(' - Cups ' + str(totalBrought(allGuests, 'cups')))
# print(' - Cakes ' + str(totalBrought(allGuests, 'cakes')))
# print(' - Ham Sandwiches ' + str(totalBrought(allGuests, 'ham sandwiches')))
# print(' - Apple Pies ' + str(totalBrought(allGuests, 'apple pies')))
# # # # ---------------------------------------------------
# # Fantasy Game Inventory
# # stuff = {'rope': 1, 'torch': 6, 'gold coin': 42, 'dagger': 1, 'arrow': 12}


# def displayInventory(stuff):
#     print('Inventory:')
#     total = 0
#     for k, v in stuff.items():
#         print(str(v) + " " + k)
#         total += v
#     print(f'Total number of items: {total}')


# # displayInventory(stuff)

# # # # # ---------------------------------------------------
# # List to Dictionary Function for Fantasy Game Inventory
# inv = {'gold coin': 42, 'rope': 1}
# dragonLoot = ['gold coin', 'dagger', 'gold coin', 'gold coin', 'ruby']


# def addToInventory(inventory, addedItems):
#     for item in addedItems:
#         # add new if not exist
#         inventory.setdefault(item, 0)
#         # increase number
#         inventory[item] += 1
#     return inventory


# inv = addToInventory(inv, dragonLoot)
# displayInventory(inv)
# # # # # ---------------------------------------------------
# # Manipulating Strings

# # String Literals
# # 'That is Alice's cat.' # wrong

# # Double Quotes
# # spam = "That is Alice's cat."

# # # Escape Characters
# # spam = 'Say hi to Bob\'s mother.'
# # print(spam)

# # # Raw Strings
# # print(r'That is Carol\'s cat.')
# # print('That is Carol\'s cat.')

# # # Multiline Strings with Triple Quotes
# # print('''Dear Alice,
# # Eve's cat has been arrested for catnapping, cat burglary, and extortion.
# # Sincerely,
# # Bob''')


# # Multiline Comments

# # Indexing and Slicing Strings

# # in not in

# # upper(), lower(), isupper(), and islower() String Methods

# # The isX String Methods

# # startswith() and endswith() String Methods

# # join() and split() String Methods
# print(', '.join(['cats', 'rats', 'bats']))
# print('My name is Simon'.split('m'))    # whitespace default

# spam = '''Dear Alice,
# How have you been? I am fine.
# There is a container in the fridge
# that is labeled "Milk Experiment".
# Please do not drink it.
# Sincerely,
# Bob'''

# print(spam.split('\n'))
# # # # # ---------------------------------------------------

# Justifying Text with rjust(), ljust(), and center()

# Removing Whitespace with strip(), rstrip(), and lstrip()

# Copying and Pasting Strings with the pyperclip Module
# # # # # ---------------------------------------------------
# page 136

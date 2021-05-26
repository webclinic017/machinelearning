from github import Github

ACCESS_TOKEN = open('token.txt', 'r').read()

g = Github(ACCESS_TOKEN)
user = g.get_user()
login = user.login
# print(user)
# print(login)
query = 'language:python'
result = g.search_repositories(query)

for item in result:
    print(item)

#https://github.com/Voyz/ibind
from ibind import IbkrClient

# Construct the client
client = IbkrClient(port='8888')

# Call some endpoints
print('\n#### check_health ####')
print(client.check_health())

print('\n\n#### tickle ####')
print(client.tickle().data)

print('\n\n#### get_accounts ####')
print(client.portfolio_accounts().data)


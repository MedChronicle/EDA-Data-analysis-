class Animal:
    def __init__(self, name, sound):
        self.name = name
        self.sound = sound

    def make_sound(self):
        sound_message = f"{self.name} says {self.sound}"
        return sound_message

class Dog(Animal):
    def __init__(self, name, breed):
        dog_sound = "Woof"
        super().__init__(name, dog_sound)
        self.breed = breed

    def fetch(self):
        fetch_message = f"{self.name} the {self.breed} fetches the ball"
        return fetch_message

class Cat(Animal):
    def __init__(self, name):
        cat_sound = "Meow"
        super().__init__(name, cat_sound)

dog_name = "Rex"
dog_breed = "Labrador"
cat_name = "Whiskers"

dog = Dog(dog_name, dog_breed)
cat = Cat(cat_name)

dog_sound_message = dog.make_sound()
dog_fetch_message = dog.fetch()
cat_sound_message = cat.make_sound()

print(dog_sound_message)
print(dog_fetch_message)
print(cat_sound_message)

dog_is_animal = isinstance(dog, Animal)
dog_class_is_subclass = issubclass(Dog, Animal)

print(dog_is_animal)
print(dog_class_is_subclass)

class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount > self.balance:
            insufficient_message = "Insufficient funds"
            print(insufficient_message)
        else:
            self.balance -= amount

account_owner = "Dana"
starting_balance = 100
deposit_amount = 50
withdraw_amount = 30

account = BankAccount(account_owner, starting_balance)
account.deposit(deposit_amount)
account.withdraw(withdraw_amount)

final_balance = account.balance
print(final_balance)

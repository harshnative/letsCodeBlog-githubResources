from passwords import GlobalData


# function to check if the password is correct or not
def passwordChecker_func(password):

    lenPass = len(password)

    if(lenPass != GlobalData.lenMyPass):
        return False
    

    for i in range(GlobalData.lenMyPass):
        if(GlobalData.myPass[i] != password[i]):
            return False

    return True
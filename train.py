from modules.MySES import MySES

if __name__ == "__main__":
    model = MySES()
    # model.train()
    model.train_one(1)
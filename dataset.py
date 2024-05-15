# import some packages you need here

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):

        # write your codes here

    def __len__(self):

        # write your codes here

    def __getitem__(self, idx):

        # write your codes here

        return input, target

if __name__ == '__main__':

    # write test codes to verify your implementations
'''
    DS 5230
    Summer 2022
    HW1_Problem_2_Kosarak_Association_Rules

    Transform input data's format to a specific tabular format that is 
    required by Weka

    Hongyan Yang
'''


import time

def generate_attributes_set(path):
    '''
    Function -- generate_attributes_set
    Generate a list of attributes from the input file
    Parameters: path (str)  -- input file's path    
    Return a sorted list of all attributes from input data
    '''
    attributes_set = set()
    with open(path, encoding = "utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            attributes = line.split(" ")
            # Transform str to int in order to sort data
            attributes = [int(each) for each in attributes]
            # Apply set operations to remove redundant data
            attributes_set.update(attributes)
    return sorted(list(attributes_set))

def generate_instances(path):
    '''
    Function -- generate_instances
    Transfrom each instance to the format required by a sparse
    kosarak.arff file
    Parameters: path (str)  -- input file's path    
    Return a list of all instances in transformed format
    '''
    instances_list = []
    with open(path, encoding = "utf-8") as f:
        for line in f:
            line_set = set()
            line = line.rstrip("\n")
            line_attri = line.split(" ")
            line_attri = [int(each) for each in line_attri]
            line_set.update(line_attri)
            line_attri = sorted(list(line_set))
            # Record each instance in sparse .arff file's format
            temp_str = "{"
            for each in line_attri:
                temp_str += str(each - 1) + " 1,"
            temp_str = temp_str.rstrip(",")
            temp_str += "}"
            instances_list.append(temp_str)
    return instances_list

def write_arff(path):
    '''
    Function -- write_arff
    Transform the input file to a sparse ARFF file
    Parameters: path (str)  -- input file's path    
    Create a sparse .arff file
    '''
    attributes = generate_attributes_set(path)
    instances_list = generate_instances(path)
    # Create the output file's name
    out_path = path.replace(".dat", ".arff")
    with open(out_path, "a", encoding = "utf-8") as f:
        f.write("@RELATION Kosarak\n")
        # Write attributes
        for each in attributes:
            temp_str = "@ATTRIBUTE news_item_" + str(each) + " {0, 1}\n"
            f.write(temp_str)
        f.write("@DATA\n")
        # Write data
        for each in instances_list:
            f.write(each + "\n") 

def main():
    begin = time.time()
    write_arff("kosarak.dat")
    end = time.time()
    print("Total runtime of the program is %.4fs." % (end - begin))


if __name__ == "__main__":
    main()

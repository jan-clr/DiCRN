
class Reaction:

    def __init__(self, reactants, products, catalyst=None, k_f=1.0, k_b=1.0, k_cat=1.0, k_m=1.0, stoichiometry=None):
        """
        A reaction is a process that transforms reactants into products with a catalyst present or not.

        :param reactants: An array of reactants encoded as integers which represent the species in the CRN.
        :param products: An array of products encoded as integers which represent the species in the CRN.
        :param catalyst: A catalyst encoded as an integer which represents the species in the CRN. None if there is no catalyst.
        :param rate: A float that represents the rate of the reaction in M/s.
        :param stoichiometry: An array of integers that represent the stoichiometry of the reaction.
        """
        if stoichiometry is None:
            stoichiometry = [1, 1, 1]
        self.reactants = reactants
        self.products = products
        self.catalyst = catalyst
        self.stoichiometry = stoichiometry
        self.k_f = k_f
        self.k_b = k_b
        self.k_cat = k_cat
        self.k_m = k_m

    @staticmethod
    def from_identifier(reaction_str):
        """
        Create a Reaction object from a string that is encoded like in this paper
        https://pmc.ncbi.nlm.nih.gov/articles/PMC2440819/#abstract2
        There are twelve atom reaction classes in the model,
        each of which is represented by a letter, while a unique reaction is represented by a 4-letter identifier string.
        The first letter represents the reaction class, the second to fourth letters represent the reactants, with the
        fourth letter being 'X' if there is no third reactant.

        >>> Reaction.from_identifier("Jabc")
        Will create a catalytic reaction with reactant a, catalyst b and product c. (And default rate 1.0)
        The integer identifier of the species are 0-indexed positions in the alphabet. (a=0, b=1, c=2)

        :param reaction_str: A string that represents a reaction.
        :return: A Reaction object.
        """
        reaction_class, reactant1, reactant2, reactant3 = list(reaction_str)
        match reaction_class:
            case "A":
                if reactant3 != 'X':
                    raise ValueError("Invalid reaction string. Class A reactions should have 2 reactants.")
                return Reaction([char_to_int(reactant1)], [char_to_int(reactant2)], None, stoichiometry=[-1, 1, 0])
            case "B":
                return Reaction([char_to_int(reactant1)], [char_to_int(reactant2)], None, stoichiometry=[-2, 1, 0])
            case "C":
                return Reaction([char_to_int(reactant1)], [char_to_int(reactant2)], char_to_int(reactant1), stoichiometry=[-1, 1, 0])
            case "D":
                return Reaction([char_to_int(reactant1)], [char_to_int(reactant2)], char_to_int(reactant2), stoichiometry=[-1, 1, 0])
            case "E":
                return Reaction([char_to_int(reactant1)], [char_to_int(reactant2), char_to_int(reactant3)], None, stoichiometry=[-1, 1, 1])
            case "F":
                return Reaction([char_to_int(reactant1)], [char_to_int(reactant2), char_to_int(reactant3)], None, stoichiometry=[-2, 1, 1])
            case "G":
                return Reaction([char_to_int(reactant1), char_to_int(reactant2)], [char_to_int(reactant3)], None, stoichiometry=[-2, -1, 1])
            case "H":
                return Reaction([char_to_int(reactant1), char_to_int(reactant2)], [char_to_int(reactant3)], None, stoichiometry=[-2, -1, 2])
            case "I":
                return Reaction([char_to_int(reactant1), char_to_int(reactant2)], [char_to_int(reactant3)], None, stoichiometry=[-4, -1, 1])
            case "J":
                return Reaction([char_to_int(reactant1)], [char_to_int(reactant3)], char_to_int(reactant2), stoichiometry=[-1, 1, 1])
            case "K":
                return Reaction([char_to_int(reactant1)], [char_to_int(reactant2), char_to_int(reactant3)], char_to_int(reactant1), stoichiometry=[-1, 1, 1])
            case "L":
                return Reaction([char_to_int(reactant1)], [char_to_int(reactant2), char_to_int(reactant3)], char_to_int(reactant2), stoichiometry=[-1, 1, 1])

    def to_identifier(self):
        """
        Convert a Reaction object to a 4-letter identifier string. With the first letter representing the reaction class,
        the second to fourth letters representing the reactants, with the fourth letter being 'X' if there is no third reactant according to the paper.
        :return: A 4 letter identifier string.
        """
        species = self.reactants + self.products
        species_string = ''.join([int_to_char(specie) for specie in species])
        if self.catalyst is None:
            if len(self.reactants) == 1:
                if self.stoichiometry == [-1, 1, 0]:
                    return f"A{species_string}X"
                elif self.stoichiometry == [-2, 1, 0]:
                    return f"B{species_string}X"
                elif self.stoichiometry == [-1, 1, 1]:
                    return f"E{species_string}"
                elif self.stoichiometry == [-2, 1, 1]:
                    return f"F{species_string}"
            elif len(self.reactants) == 2:
                if self.stoichiometry == [-2, -1, 1]:
                    return f"G{species_string}"
                elif self.stoichiometry == [-2, -1, 2]:
                    return f"H{species_string}"
                elif self.stoichiometry == [-4, -1, 1]:
                    return f"I{species_string}"
        else:
            if len(species) == 2:
                if self.catalyst == species[0]:
                    return f"C{species_string}X"
                elif self.catalyst == species[1]:
                    return f"D{species_string}X"
                else:
                    return f"J{species_string[0]}{int_to_char(self.catalyst)}{species_string[1]}"
            elif len(species) == 3:
                if self.catalyst == species[0]:
                    return f"K{species_string}"
                elif self.catalyst == species[1]:
                    return f"L{species_string}"

    def __repr__(self):
        formula = ' + '.join([(f"{abs(self.stoichiometry[i])}" if abs(self.stoichiometry[i]) != 1 else '') + int_to_char(reactant) for i, reactant in enumerate(self.reactants)])
        formula += " <--> " if self.catalyst is None else f" \u2500\u2500{int_to_char(self.catalyst)}-> "
        nr_reactants = len(self.reactants)
        formula += ' + '.join([f"{abs(self.stoichiometry[i + nr_reactants])}" if abs(self.stoichiometry[i + nr_reactants]) != 1 else '' + int_to_char(product) for i, product in enumerate(self.products)])
        return formula

    def get_species(self):
        return self.reactants + self.products + ([self.catalyst] if self.catalyst is not None else [])


def char_to_int(char):
    return ord(char) - ord('a')


def int_to_char(integer):
    return chr(integer + ord('a'))


def main():
    r = Reaction.from_identifier("Ladb")
    print(r)
    print(r.to_identifier())


if __name__ == '__main__':
    main()

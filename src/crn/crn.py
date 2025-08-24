from torch_geometric.data.remote_backend_utils import num_nodes

from src.crn.reaction import Reaction, int_to_char
from mass import MassReaction, MassModel, MassMetabolite
from torch_geometric.data import Data
import torch


class CRN:

    def __init__(self, reactions, initial_concentrations=None):
        self.reactions = reactions

        # Find all species in the reactions
        species = set()
        for reaction in reactions:
            species.update(reaction.get_species())
        species = sorted(list(species))
        if len(species) == 0:
            raise ValueError("No species found in the reactions.")
        if species[-1] != len(species) - 1:
            raise ValueError("Species indices should not skip any available range.")
        self.nr_species = len(species)

        if initial_concentrations is None:
            initial_concentrations = [1.0] * self.nr_species
        elif len(initial_concentrations) != self.nr_species:
            raise ValueError("Initial concentrations should have the same length as the number of species.")

        self.initial_concentrations = initial_concentrations
        self.mass_metabolites = []
        self.mass_reactions = []
        self.mass_model = None
        self.mass_metabolite_keys = []

    @staticmethod
    def from_signature(signature):
        reaction_strings = signature.split("_")
        reactions = []
        for reaction_string in reaction_strings:
            reaction = Reaction.from_identifier(reaction_string)
            reactions.append(reaction)

        return CRN(reactions)

    def __repr__(self):
        return "\n".join([str(reaction) for reaction in self.reactions])

    def to_signature(self):
        return "_".join([reaction.to_identifier() for reaction in self.reactions])

    def to_mass_model(self):
        self.mass_metabolites = [MassMetabolite(id_or_specie=int_to_char(i), name=f"species_{int_to_char(i)}", compartment="compartment") for i in range(self.nr_species)]
        for i, metabolite in enumerate(self.mass_metabolites):
            metabolite.ic = self.initial_concentrations[i]
        self.mass_metabolite_keys = [met.id for met in self.mass_metabolites]
        self.mass_reactions = []
        for reaction in self.reactions:
            if reaction.catalyst is None:
                mr1 = MassReaction('reaction_' + reaction.to_identifier())
                mr1.reversible = False
                mr1.forward_rate_constant = reaction.k_f
                mr1.add_metabolites({self.mass_metabolites[reactant]: reaction.stoichiometry[i] for i, reactant in enumerate(reaction.reactants)})
                mr1.add_metabolites({self.mass_metabolites[product]: reaction.stoichiometry[i + len(reaction.reactants)] for i, product in enumerate(reaction.products)})
                self.mass_reactions.append(mr1)
                mr2 = MassReaction('reaction_' + reaction.to_identifier() + "_reverse")
                mr2.reversible = False
                mr2.forward_rate_constant = reaction.k_b
                mr2.add_metabolites({self.mass_metabolites[product]: -reaction.stoichiometry[i + len(reaction.reactants)] for i, product in enumerate(reaction.products)})
                mr2.add_metabolites({self.mass_metabolites[reactant]: -reaction.stoichiometry[i] for i, reactant in enumerate(reaction.reactants)})
                self.mass_reactions.append(mr2)
            else:
                # substrate enzyme binding
                mr1 = MassReaction('reaction_' + reaction.to_identifier() + "_binding")
                mr1.reversible = False
                mr1.forward_rate_constant = reaction.k_cat / reaction.k_m
                mr1.add_metabolites({self.mass_metabolites[reaction.reactants[0]]: -1})
                mr1.add_metabolites({self.mass_metabolites[reaction.catalyst]: -1})
                es_key = f"{int_to_char(reaction.catalyst)}_{int_to_char(reaction.reactants[0])}"
                es_complex = None
                if es_key not in self.mass_metabolite_keys:
                    es_complex = MassMetabolite(id_or_specie=es_key, name=f"complex_{es_key}", compartment="compartment")
                    es_complex.ic = 0.0
                    self.mass_metabolites.append(es_complex)
                    self.mass_metabolite_keys.append(es_key)
                else:
                    es_complex = self.mass_metabolites[self.mass_metabolite_keys.index(es_key)]

                mr1.add_metabolites({es_complex: 1})

                # catalysis
                mr2 = MassReaction('reaction_' + reaction.to_identifier() + "_product_formation")
                mr2.reversible = False
                mr2.forward_rate_constant = reaction.k_cat
                mr2.add_metabolites({es_complex: -1})
                if reaction.catalyst == reaction.reactants[0]:
                    mr2.add_metabolites({self.mass_metabolites[reaction.products[0]]: reaction.stoichiometry[1] + 1})
                else:
                    mr2.add_metabolites({self.mass_metabolites[reaction.products[0]]: reaction.stoichiometry[1]})
                    mr2.add_metabolites({self.mass_metabolites[reaction.catalyst]: 1})

                if len(reaction.products) == 2:
                    mr2.add_metabolites({self.mass_metabolites[reaction.products[1]]: reaction.stoichiometry[2]})

                self.mass_reactions.append(mr1)
                self.mass_reactions.append(mr2)

        self.mass_model = MassModel(self.to_signature())
        self.mass_model.add_reactions(self.mass_reactions)
        return self.mass_model

    def get_odes(self):
        self.to_mass_model()
        return {met.id : ode for met, ode in self.mass_model.odes.items() if met.id in self.mass_metabolite_keys}

    def stoichiometry_matrix(self):
        self.to_mass_model()
        return self.mass_model.S

    def to_polynomials(self):
        S = self.stoichiometry_matrix()
        pass

    def to_graph(self):
        S = self.stoichiometry_matrix()
        metabolites = [met.id for met in self.mass_model.metabolites]
        edge_index = []
        edge_attr = []
        # The graph has as many nodes as the number of species plus the number of reactions
        nr_species = S.shape[0]
        nr_reactions = S.shape[1]
        nr_nodes = nr_species + nr_reactions
        # define the first nr_species nodes as species nodes and the rest as reaction nodes
        for s in range(nr_species):
            for r in range(nr_reactions):
                if S[s, r] < 0:
                    edge_index.append([s, r + nr_species])
                    edge_attr.append([-S[s, r]])
                elif S[s, r] > 0:
                    edge_index.append([r + nr_species, s])
                    edge_attr.append([S[s, r]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).T.contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        return Data(edge_index=edge_index, edge_attr=edge_attr, species_names=metabolites, num_nodes=nr_nodes, num_species=nr_species, num_reactions=nr_reactions, signature=self.to_signature())


def main():
    crn = CRN.from_signature("DabX_Jbca")
    print(crn)
    print(crn.to_signature())
    model = crn.to_mass_model()
    print(model.S)


if __name__ == '__main__':
    main()
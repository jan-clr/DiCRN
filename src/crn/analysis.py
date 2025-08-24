
from juliacall import Main as jl, convert as jlconvert
from mass import MassModel
import numpy as np
from pathlib import Path

from src.crn import CRN
from src.helpers.crn_analyis import calculate_preserved_moieties, sort_by_species_name


def analyse_crn(crn: CRN, jl_main, nr_models=100, nr_initial_concentration_sets=10):
    model: MassModel = crn.to_mass_model()

    T, P = calculate_preserved_moieties(model.S)
    ids = model.metabolites.list_attr("id")

    S, T, P, ids = sort_by_species_name(model.S, T, P, ids)

    # Split S into A and B, where A is a matrix with the same shape as S but only with negative entries of S and B is the same but with positive entries
    A = np.where(S < 0, -S, 0).T.astype(int)

    # Use S, A and T to calculate coefficients for the polynomials to use in the homotopy continuation.
    # For each polynomial we need a matrix with the exponent of each species in all monomials and a vector with the constant term of each monomial.

    # First we calculate the exponent matrix for each polynomial. We have as many polynomials as independent species.
    # The exponent matrix is a matrix with the same number of rows as variables (species) and the same number of columns as monomials in the polynomial.
    # Since we always have the same monomials available, this matrix is the same for all polynomials and is just A.T.

    # The coefficients for each monomial are the rows of S

    jl.seval('using HomotopyContinuation, LinearAlgebra, Random')
    jl.seval('using .CRNAnalysis')

    F = jl.construct_system_from_matrices(A.T, S.astype(int), T)
    print(F)
    
    nr_dependent_species = len(T)
    nr_independent_species = len(S) - nr_dependent_species
    
    nr_models = 100
    nr_initial_concentration_sets = 10
    
    n_0 = [1.0] * nr_independent_species
    k_0 = [1.0] * S.shape[1]
    
    data_points = jl.find_stationary_points(F, jlconvert(jl.Vector, n_0), jlconvert(jl.Vector, k_0), nr_models, nr_initial_concentration_sets)
    
    model_solutions = jl.analyze_stability(F, data_points, nr_initial_concentration_sets, len(n_0), len(k_0))

    return model_solutions


def analyse_crn_from_signature(signature, jl_main, nr_models=100, nr_initial_concentration_sets=10):
    crn = CRN.from_signature(signature)
    return analyse_crn(crn, jl_main, nr_models, nr_initial_concentration_sets)


def instantiate_julia():
    # Get the directory containing this file
    current_dir = Path(__file__).parent

    # Define the path to the file you want to load
    julia_src_file = current_dir / "CRNAnalysis.jl"

    jl.include(str(julia_src_file))
    return jl
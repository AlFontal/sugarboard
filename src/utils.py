def mean_glucose_to_hba1c(g: float) -> float:
    """
    Computes the estimated Hba1C (or GMI) as defined in Bergenstal et al (doi: 10.2337/dc18-1581)
    
    Parameters
    ----------
    g : float
    Mean glucose from the CGM readings
    
    Returns
    -------
    float
    Estimated Hba1c (in %).
    """
    return 3.31 + 0.02395 * g

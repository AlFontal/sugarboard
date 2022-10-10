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


def inject_align_style(text: str, alignment='center') -> str:
    """
    Injects the text within a style element with the specified alignment
    
    Parameters
    ----------
    text : str
    Text to be aligned
    
    Returns
    -------
    str
    Text with the style injected
    """
    header_level = text.count('#')
    text = text.replace('#', '')
    return f"<h{header_level} style='text-align: {alignment}'>{text}</h{header_level}>"



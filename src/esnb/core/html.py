def gen_html_sub():
    """
    Generates a string containing HTML and CSS for a styled table.

    Returns
    -------
    str
        A string with embedded CSS styles for a table with the class
        'cool-class-table'. The styles include border, padding, alternating row
        colors, and custom styles for <summary> and <details> elements.

    Examples
    --------
    >>> html = gen_html_sub()
    >>> print(html)
    """
    html = """
        <style>
            .cool-class-table {{
                border-collapse: collapse;
                width: 400px;
                font-family: sans-serif;
                text-align: left;
            }}
            .cool-class-table th, .cool-class-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left !important;
            }}
            .cool-class-table tr:nth-child(even) {{
                background-color: #f2f2f2; /* Lighter Gray */
                text-align: left;
            }}
            .cool-class-table tr:nth-child(odd) {{
                background-color: #ffffff; /* White */
                text-align: left;
            }}
            .cool-class-table summary {{
                cursor: pointer;
                font-weight: bold;
            }}
            .cool-class-table details > div {{
                padding: 10px;
                border-top: 1px solid #ccc;
                margin-top: 5px;
            }}
        </style>
        """
    return html

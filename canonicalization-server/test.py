from canon_tools import canonicalize_bigsmiles

def test():
    test_bigsmiles = "CCO{[>][<]CCO[>][<]}CCO"
    canonical = canonicalize_bigsmiles(
        test_bigsmiles,
        output_folder=None,
        plot=False
    )
    print(canonical)


if __name__ == "__main__":
    test()

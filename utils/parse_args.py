import argparse

def parse_args(arg_specs):
    parser = argparse.ArgumentParser(description="Parse command line arguments.")
    
    for arg, arg_type in arg_specs.items():
        if arg_type[0] == bool:
            parser.add_argument(f'--{arg}', action=argparse.BooleanOptionalAction, help=f'{arg} argument', default=arg_type[1])
        else:
            parser.add_argument(f'--{arg}', type=arg_type[0], help=f'{arg} argument', default=arg_type[1])
    
    args = parser.parse_args()
    return vars(args)
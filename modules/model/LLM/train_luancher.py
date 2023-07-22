import accelerate.commands.accelerate_cli as acli
import os


def main():
    """
    Main function to start the training with accelerate config file loaded.

    Returns:
        None
    """
    parser = acli.ArgumentParser("Accelerate CLI tool", usage="accelerate <command> [<args>]")
    subparsers = parser.add_subparsers(help="accelerate command helpers")

    # Register commands
    acli.get_config_parser(subparsers=subparsers)
    acli.env_command_parser(subparsers=subparsers)
    acli.launch_command_parser(subparsers=subparsers)
    acli.tpu_command_parser(subparsers=subparsers)
    acli.test_command_parser(subparsers=subparsers)

    #
    # current_dir = os.getcwd()
    # file_path = os.path.join(current_dir, "train.py")

    real_path = os.path.split(os.path.realpath(__file__))[0]
    new_path = os.path.join(real_path, "..","..", "..","data", "config", "train_config", "accelerate_config.yaml")
    new_path2 = os.path.join(real_path, "train.py")

    args = parser.parse_args(["launch", "--config_file", new_path, new_path2])

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    args.func(args)




# if __name__=='__main__':
#     main()
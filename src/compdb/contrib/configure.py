#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

from os.path import expanduser

RE_EMAIL = r"[^@]+@[^@]+\.[^@]+"

OPERATIONS= ['add', 'set', 'remove', 'dump', 'show']
USER_GLOBAL = expanduser('~/compdb.rc')
USER_LOCAL = expanduser('./compdb.rc')

def process(args):
    from compdb.core.config import DIRS, FILES
    from os.path import abspath, expanduser
    if args.name: 
        if args.name in DIRS or args.name in FILES:
            args.value = abspath(expanduser(args.value))

def get_config(args, for_writing = False):
    from compdb.core.config import Config, load_config
    config = Config()
    try:
        if args._global:
            config.read(USER_GLOBAL)
        elif args.config:
            config.read(args.config)
        elif for_writing:
            config.read(USER_LOCAL)
        else:
            config = load_config()
            #config.read(expanduser('./compdb.rc'))
    except FileNotFoundError:
        pass
    return config

def write_config(config, args):
    if args._global:
        config.write(USER_GLOBAL)
    elif args.config == '-':
        config.dump()
    elif args.config:
        config.write(args.config)
    else:
        config.write(USER_LOCAL)
        #msg = "You need to use option '--global' or '--config' to specify which config file to write to."
        #raise ValueError(msg)

def check_name(args):
    from ..core.config import LEGAL_ARGS
    if not args.force:
        if not args.name in LEGAL_ARGS:
            msg = "'{}' does not seem to be a valid configuration value. Use '-f' or '--force' to ignore this warning."
            raise ValueError(msg.format(args.name))

def add(args):
    check_name(args)
    config = get_config(args, for_writing = True)
    if args.name in config:
        msg = "Value for '{}' is already set in '{}'. Use 'set' instead of 'add' to overwrite."
        print(msg.format(args.name, args.config))
        return
    config[args.name] = args.value
    write_config(config, args)

def set_value(args):
    check_name(args)
    config = get_config(args, for_writing = True)
    config[args.name] = args.value
    write_config(config, args)

def remove(args):
    config = get_config(args, for_writing = True)
    del config[args.name]
    write_config(config, args)

def dump(args):
    config = get_config(args)
    config.dump(indent = 1)

def show(args):
    from ..core.config import LEGAL_ARGS, DEFAULTS
    config = get_config(args)
    legal_args = sorted(LEGAL_ARGS)
    l_column0 = max(len(arg) for arg in legal_args)
    print("Current configuration:")
    print()
    msg = "{arg:<" + str(l_column0) + "}: {value}"
    for arg in legal_args:
        line = msg.format(arg = arg, value = config.get(arg))
        print(line)

def verify(args):
    import re
    args.name = args.name.strip()
    args.email = args.email.strip()
    if not re.match(RE_EMAIL, args.email):
        msg = "Invalid email address: '{}'."
        raise ValueError(msg.format(args.email))
    if args.config != '-':
        from os.path import expanduser, realpath
        args.config = realpath(expanduser(args.config))

#def make_author(args):
#    from compdb.core.config import Config
#    import os
#    c = {
#        'author_name': args.name,
#        'author_email': args.email,
#    }
#    config = Config()
#    if not args.config == '-':
#        try:
#            config.read(args.config)
#        except FileNotFoundError:
#            pass
#    config.update(c)
#    if args.config == '-':
#        config.dump()
#    else:
#        config.write(args.config)

def configure(args):
    process(args)
    if args.operation == 'add':
        add(args)
    elif args.operation == 'set':
        set_value(args)
    elif args.operation == 'remove':
        remove(args)
    elif args.operation == 'dump':
        dump(args)
    elif args.operation == 'show':
        show(args)
    else:
        print("Unknown operation: {}".format(args.operation))

HELP_OPERATION = """\
    R|Configure compdb for your local environment.
    You can perform one of the following operations:
        
        set:    Set value of 'name' to 'value'.

        add:    Like 'set', but will not overwrite
                any existing values.

        remove: Remove configuration value 'name'.

        dump:   Dump the selected configuration.

        show:   Show the complete configuration
                including default values.

    """
import textwrap

def setup_parser(parser):
        parser.add_argument(
            'operation',
            type = str,
            choices = OPERATIONS,
            help = textwrap.dedent(HELP_OPERATION))
        parser.add_argument(
            'name',
            type = str,
            nargs = '?',
            help = "variable name")
        parser.add_argument(
            'value',
            type = str,
            nargs = '?',
            default = '',
            help = "variable value")
        parser.add_argument(
            '-c', '--config',
            type = str,
            #default = expanduser('./compdb.rc'),
            help = "The config file to read and write from. Use '-' to print to standard output.")
        parser.add_argument(
            '-g', '--global',
            dest = '_global',
            action = 'store_true',
            help = "Write to the user's global configuration file.")
        parser.add_argument(
            '-f', '--force',
            action = 'store_true',
            help = "Ignore all warnings.")

def main(arguments = None):
        from argparse import ArgumentParser
        parser = ArgumentParser(
            description = "Change the compDB configuration.",
            )
        args = parser.parse_args(arguments)
        return configure(args)

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO)
    main()
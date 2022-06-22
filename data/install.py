from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
import os
from torchvision.datasets.cifar import CIFAR10
import subprocess
import argparse
import json

parser = argparse.ArgumentParser(description='Download dataset')
parser.add_argument('-y', '--yes', action='store_true', help='Answer yes to all questions', default=False)
args = parser.parse_args()

ROOT = os.environ['DATA_ROOT']


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def read_or_create(path):
    if os.path.exists(path):
        return json.load(open(path, 'r'))
    else:
        return {}


def print_and_run(args):
    print('Running Command:', bcolors.OKBLUE, ' '.join(args), bcolors.ENDC)
    subprocess.call(args)


def download_camelyon17():
    print(bcolors.HEADER, 'Downloading Camelyon17', bcolors.ENDC)
    Camelyon17Dataset(root_dir=ROOT, download=True)
    rec = read_or_create('data/paths.json')
    rec['camelyon17'] = ROOT
    json.dump(rec, open('data/paths.json', 'w'))


def download_cifar10():
    print(bcolors.HEADER, 'Downloading CIFAR10', bcolors.ENDC)
    CIFAR10(root=ROOT, train=True, download=True)
    print_and_run(['rm', '-rf', os.path.join(ROOT, 'cifar-10-python.tar.gz')])
    rec = read_or_create('data/paths.json')
    rec['cifar10'] = ROOT
    json.dump(rec, open('data/paths.json', 'w'))


def download_cifar10C():
    print(bcolors.HEADER, 'Downloading CIFAR10C', bcolors.ENDC)
    url = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1'
    # install with wget
    print_and_run(['wget', url, '-O', 'CIFAR-10-C.tar'])
    # extract
    print_and_run(['tar', '-xvf', 'CIFAR-10-C.tar'])
    # move to data folder
    print_and_run(['mv', 'CIFAR-10-C', ROOT])
    # cleanup
    print_and_run(['rm', 'CIFAR-10-C.tar'])

    rec = read_or_create('data/paths.json')
    rec['cifar10C'] = ROOT
    json.dump(rec, open('data/paths.json', 'w'))


def copy_uci_data():
    print(bcolors.HEADER, 'Copying UCI data to DATA_ROOT', bcolors.ENDC)
    print_and_run(['mkdir', os.path.join(ROOT, 'UCI')])
    print_and_run(['cp', 'data/uci_heart_torch.pt', os.path.join(ROOT, 'UCI/uci_heart_torch.pt')])

    rec = read_or_create('data/paths.json')
    rec['uci_heart'] = ROOT
    json.dump(rec, open('data/paths.json', 'w'))


def input_check_flag(prompt):
    if args.yes:
        print(bcolors.OKGREEN + prompt + bcolors.ENDC, 'y')
        return True
    else:
        return input(bcolors.OKGREEN + prompt + bcolors.ENDC).lower() == 'y'


def run_if_y(prompt, func, else_message=None):
    if input_check_flag(prompt):
        func()
    else:
        if else_message is not None:
            print(bcolors.WARNING, else_message, bcolors.ENDC)


if __name__ == '__main__':

    res = input_check_flag(f'Data root set to ({bcolors.BOLD}{ROOT}{bcolors.ENDC}). Confirm (y/n): ')
    if res:
        run_if_y('Download CIFAR-10? (y/n): ', download_cifar10,
                 else_message='Skipping CIFAR-10')
        run_if_y('Download CIFAR-10C? (y/n): ', download_cifar10C, else_message='Skipping CIFAR-10C')
        run_if_y('Download Camelyon17? (y/n): ', download_camelyon17, else_message='Skipping Camelyon17')
        run_if_y('Download UCI Heart? (y/n): ', copy_uci_data, else_message='Skipping UCI Heart')
    else:
        print(bcolors.WARNING + 'Aborting' + bcolors.ENDC)
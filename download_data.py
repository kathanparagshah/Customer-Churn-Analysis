#!/usr/bin/env python3

from src.data.load_data import DataLoader

def main():
    loader = DataLoader()
    file_path = loader.download_from_kaggle()
    print(f'Downloaded to: {file_path}')
    print(f'File exists: {file_path.exists()}')

if __name__ == '__main__':
    main()
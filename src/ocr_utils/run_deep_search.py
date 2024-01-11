import deepsearch as ds
import argparse


def parse_arguments():
    argparser = argparse.ArgumentParser(description="Prepare the image-data set for training or prediction")
    argparser.add_argument('--source-path', type=str, required=False,
                           default="/dccstor/alfassy/dev/open_clip_honda/ccstest.zip",
                           help="path to input documents as explained here: https://github.com/DS4SD/deepsearch-examples/blob/main/examples/document_conversion/notebooks/convert_documents.ipynb")
    argparser.add_argument('--target-dir', type=str, required=False, default="../data/pdf_data",
                           help="path to directory to store output annotations' zip files")
    argparser.add_argument('--username', type=str, default="amit.alfassy@ibm.com",
                           help="DeepSearch username as explained here: https://ds4sd.github.io/deepsearch-toolkit/getting_started/")
    argparser.add_argument('--api_key', type=str, default="yourapikey",
                           help="DeepSearch API key as explained here: https://ds4sd.github.io/deepsearch-toolkit/getting_started/")
    args = argparser.parse_args()
    return args


def main():
    args = parse_arguments()
    host = "https://deepsearch-experience.res.ibm.com"
    proj = "yourprojnumber"
    username = args.username
    api_key = args.api_key
    auth = ds.DeepSearchKeyAuth(username=username, api_key=api_key)
    config = ds.DeepSearchConfig(host=host, auth=auth)
    client = ds.CpsApiClient(config)
    api = ds.CpsApi(client)

    documents = ds.convert_documents(
        api=api,
        proj_key=proj,
        source_path=args.source_path,
        progress_bar=True
    )
    documents.download_all(result_dir=args.target_dir, progress_bar=True)
    info = documents.generate_report(result_dir=args.target_dir)
    print(info)


if __name__ == '__main__':
    main()

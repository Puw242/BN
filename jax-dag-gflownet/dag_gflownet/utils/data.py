import pandas as pd
import urllib.request
import gzip

from pathlib import Path
from numpy.random import default_rng
from pgmpy.utils import get_example_model

from dag_gflownet.utils.graph import sample_erdos_renyi_linear_gaussian
from dag_gflownet.utils.sampling import sample_from_linear_gaussian


def download(url, filename):
    if filename.is_file():
        return filename
    filename.parent.mkdir(exist_ok=True)

    # Download & uncompress archive
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as uncompressed:
            file_content = uncompressed.read()

    with open(filename, "wb") as f:
        f.write(file_content)

    return filename


def get_data(name, args, rng=default_rng()):
    graph = sample_erdos_renyi_linear_gaussian(
        num_variables=args.num_variables,
        num_edges=args.num_edges,
        loc_edges=0.0,
        scale_edges=1.0,
        obs_noise=0.1,
        rng=rng,
    )
    score = "bge"
    data_all = pd.DataFrame()
    for i in range(1, 13):
        t = pd.read_csv(
                "../data/digit/wa_2022_"
            + str(i)
            + ".csv"
        )
        data_all = pd.concat([data_all, t], axis=0)
    userd_attr = [
        "TOT_INJ",
        "REST1_0",
        "ALCFLAG",
        "MEDCAUSE",
        "SPEED_LIMIT",
        # "CITY",
        "V1CMPDIR",
        "V1EVENT1",
        "V2CMPDIR",
        "V2EVENT1",
        "NUMVEHS",
        "ST_FUNC",
        # "V1DIRCDE",
        "ACCTYPE",
        # "V2DIRCDE",
        "SEVERITY",
        "RDSURF",
        "LOC_TYPE",
        # "RODWYCLS",
        # "REST1_1",
        "ROUTE_MILEPOST",
        "LIGHT",
        "TIME",
        "WEEKDAY",
        "NO_PEDS",
    ]
    data_all = data_all.loc[:, userd_attr]
    print(data_all)
    data = data_all
    # if name == 'erdos_renyi_lingauss':
    #     graph = sample_erdos_renyi_linear_gaussian(
    #         num_variables=args.num_variables,
    #         num_edges=args.num_edges,
    #         loc_edges=0.0,
    #         scale_edges=1.0,
    #         obs_noise=0.1,
    #         rng=rng
    #     )
    #     data = sample_from_linear_gaussian(
    #         graph,
    #         num_samples=args.num_samples,
    #         rng=rng
    #     )
    #     score = 'bge'

    # elif name == 'sachs_continuous':
    #     graph = get_example_model('sachs')
    #     filename = download(
    #         'https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz',
    #         Path('data/sachs.data.txt')
    #     )
    #     data = pd.read_csv(filename, delimiter='\t', dtype=float)
    #     data = (data - data.mean()) / data.std()  # Standardize data
    #     score = 'bge'

    # elif name =='sachs_interventional':
    #     graph = get_example_model('sachs')
    #     filename = download(
    #         'https://www.bnlearn.com/book-crc/code/sachs.interventional.txt.gz',
    #         Path('data/sachs.interventional.txt')
    #     )
    #     data = pd.read_csv(filename, delimiter=' ', dtype='category')
    #     score = 'bde'

    # else:
    #     raise ValueError(f'Unknown graph type: {name}')

    return graph, data, score

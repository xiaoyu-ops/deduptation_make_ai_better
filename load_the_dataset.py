from torch.utils.data import Dataset
from datasets import load_dataset,load_from_disk
import os 
import requests
from PIL import Image
from io import BytesIO
import concurrent.futures
from tqdm import tqdm
import threading
import queue


def load_and_subset_dataset(dataset_name,subset_size,cache_dir):
    """
    args:
    dataset_name(str):数据集名称
    subset_size(int):子集大小,即保留数据样本的数量
    cache_dir(str):本地缓存目录
    """

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    ds = load_dataset(dataset_name,split="train")
    ds_subset = ds.select(range(subset_size))
    ds_subset.save_to_disk(cache_dir)
    print(f"数据集已经成功加载并保存到{cache_dir}")

def download_image(image_url, image_path):
    """
    下载单个图片并保存到本地

    Args:
        image_url (str): 图片 URL
        image_path (str): 图片保存路径
    Returns:
        bool: 成功返回 True,失败返回 False
    """
    try:
        response = requests.get(image_url, stream=True, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image_format = image.format if image.format else 'JPEG'
        # 保存时显式传入 format 参数，而非让 PIL 仅根据后缀推断
        image.save(image_path, format=image_format.upper())
        print(f"已成功下载并保存图片到 {image_path}")
        return True
    except (requests.exceptions.RequestException, OSError) as e:
        print(f"下载或保存图片失败：{e}")
        return False

def thread_download_worker(q, lock, counts, image_dir):
    """
        工作线程：不断从队列中获取任务进行下载
        q:任务队列,包含待下载的图片信息。
        lock:线程锁,用于保护共享数据的访问。
        counts:字典,用于统计下载成功、失败和跳过的数量。
        image_dir:图片保存目录。
    """
    while not q.empty():
        i, sample = q.get()
        try:
            image_url = sample['url']
            # 根据 LICENSE 字段决定图片格式，如果没有则使用 JPEG
            image_format = sample.get(('LICENSE')or'').lower()
            valid_exts = {"jpg", "jpeg", "png", "bmp", "gif", "webp"}
            if image_format not in valid_exts:
                image_format = "jpg"  # 对未知后缀统一用 jpg
            image_filename = f"image_{i}.{image_format}"
            image_path = os.path.join(image_dir, image_filename)
            
            # 检查图片是否已经存在
            if os.path.exists(image_path):
                print(f"图片 {image_path} 已经存在，跳过下载\n")
                with lock:
                    counts['skipped'] += 1
                q.task_done()
                continue

            result = download_image(image_url, image_path)
            with lock:
                if result:
                    counts['success'] += 1
                else:
                    counts['fail'] += 1
        except KeyError as e:
            print(f"下载或保存图片失败：{e}\n")
            with lock:
                counts['fail'] += 1
        finally:
            q.task_done()

def download_images_threading(dataset, image_dir, num_threads=4):
    """
    使用 threading 模块多线程下载数据集中 URL 对应的图片

    Args:
        dataset (Dataset): 数据集对象
        image_dir (str): 图片保存目录
        num_threads (int): 线程数，默认为 4
    """
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    total_images = len(dataset)
    counts = {'success': 0, 'fail': 0, 'skipped': 0}
    q = queue.Queue()

    # 将所有数据放入队列，并用 tqdm 显示排队进度
    for i, sample in enumerate(dataset):
        q.put((i, sample))
    print(f"共有 {total_images} 张图片待下载\n")

    lock = threading.Lock()
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=thread_download_worker, args=(q, lock, counts, image_dir))
        t.start()
        threads.append(t)

    # 用 tqdm 对队列任务完成情况进行监控
    with tqdm(total=total_images, desc="下载图片") as pbar:
        prev = 0
        while not q.empty():
            # 根据下载任务的完成数更新进度条
            completed = total_images - q.qsize()
            pbar.update(completed - prev)
            prev = completed
            # 休息一下
            threading.Event().wait(0.5)
        pbar.update(total_images - prev)

    # 等待队列完成
    q.join()
    for t in threads:
        t.join()

    print(f"下载完成：成功 {counts['success']} 张，失败 {counts['fail']} 张，跳过 {counts['skipped']} 张，总计 {total_images} 张")

if __name__ == "__main__":
    image_dir = "data/downloader_images_test"
    ds = load_from_disk("data/laion400m_subset")
    print(ds[0])
    download_images_threading(ds, image_dir, num_threads=8)
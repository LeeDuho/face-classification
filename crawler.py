from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=1,
    downloader_threads=4,
    storage={'root_dir': '/test'})
filters = dict(
    size='large',
    license='commercial,modify'
    )

#make iterator
image_keyword= ["수지", "아이유", "설현", "아이린", "김희선"]

for i in range(5):
    google_crawler.crawl(keyword=image_keyword[i] , filters=filters, max_num=20, file_idx_offset=i*20)

    bing_crawler = BingImageCrawler(downloader_threads=4, storage={'root_dir': './test_data'})
    bing_crawler.crawl(keyword=image_keyword[i], filters=None, max_num=20, file_idx_offset='auto')

    baidu_crawler = BaiduImageCrawler(storage={'root_dir': './test_data'})
    baidu_crawler.crawl(keyword=image_keyword[i], max_num=20, min_size=(200,200), max_size=None, offset= i *20)


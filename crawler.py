from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=1,
    downloader_threads=4,
    storage={'root_dir': './test_data/dog'})
filters = dict(
    size='large',
    license='commercial,modify'
    )

#make iterator
image_keyword= ['수지', '아이즈원 안유진', '이민정', '아이유', '김향기', '남주혁', '송강', 'nct 제노', 'nct 재현', '블랙핑크 지수', '배우 김정훈','배우 김혜성' ,'체리블렛 유주','프로미스나인 장규리']

for i in range(14):
    google_crawler.crawl(keyword=image_keyword[i] , filters=filters, max_num=20, file_idx_offset=i*20)

    bing_crawler = BingImageCrawler(downloader_threads=4, storage={'root_dir': './test_data/dog'})
    bing_crawler.crawl(keyword=image_keyword[i], filters=None, max_num=20, file_idx_offset='auto')

    baidu_crawler = BaiduImageCrawler(storage={'root_dir': './test_data/dog'})
    baidu_crawler.crawl(keyword=image_keyword[i], max_num=20, min_size=(200,200), max_size=None, offset= i *20)


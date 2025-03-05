def get_all(args):  # , show_underscore=1):
    for attr in dir(args):
        if not callable(getattr(args, attr)) and not attr.startswith("__"):
            try:
                # print( 'args.attr = ' , args.attr )
                print(attr, ': ', args.__getattribute__(attr), sep='')
            except Exception as exc:
                print('%-25s   fail ^^^^^^^^^^^^^^^' % attr)


        elif callable(getattr(args, attr)) and not attr.startswith("__"):
            #             if show_underscore and not attr.startswith("__"):
            print('%-25s   Callable ' % attr)
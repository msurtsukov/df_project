def run_notebook():
    import numpy as np
    import scipy as sp
    import pandas as pd
    from pandas.tseries.offsets import Hour
    from pandas import Timestamp

    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go
    plotly.offline.init_notebook_mode()
    from plotly.offline import iplot
    from plotly import tools

    from itertools import product

    import warnings
    warnings.filterwarnings('ignore')

    dfc  = pd.read_pickle("dfc.pickle")
    df   = pd.read_pickle("df.pickle")
    feat = pd.read_pickle("feat.pickle")
    pred = pd.read_pickle("pred.pickle")
    pred[pred < 0] = 0
    preds = []
    for i in range(6):
        p = pd.read_pickle("pred_%d.pickle" % (i+1))
        p[p < 0] = 0
        preds.append(p)

    N_clusters = 5
    clusters = pd.read_pickle("clusters.pickle")
    import pickle
    df_2d_mds = None
    with open("df_2d_mds.pickle", "rb") as f:
        df_2d_mds = pickle.load(f)
    df_2d_mds = pd.DataFrame(df_2d_mds, index=clusters.index)
    clusters_repr = [1387, 1230, 1326, 1734, 1333]

    dfcT = dfc.T.reset_index()
    predT = pred.T.reset_index()
    predsT = [x.T.reset_index() for x in preds]

    NY_ur_lat = 40.91553
    NY_ur_lon = -73.70001
    NY_ll_lat = 40.49612
    NY_ll_lon = -74.25559
    NY_ce_lat = (NY_ur_lat + NY_ll_lat) / 2. + 0.05
    NY_ce_lon = (NY_ur_lon + NY_ll_lon) / 2. + 0.05

    regions = pd.read_csv("data/regions.csv", delimiter=";")
    regions = regions.loc[dfc.columns-1, :]

    from geojson import FeatureCollection, Feature, Polygon
    features = []
    for i, r in regions.iterrows():
        p = Polygon([[(r["east"], r["north"]), (r["east"], r["south"]), (r["west"], r["south"]), (r["west"], r["north"]), (r["east"], r["north"])]])
        f = Feature(id=int(r["region"]), geometry=p)
        features.append(f)
    fc = FeatureCollection(features)

    import folium

    from ipywidgets import widgets
    from IPython.display import display
    from IPython.core.display import HTML
    from IPython.display import clear_output

    def inline_map(data, time, width=980, height=600):
        """Takes a folium instance and embed HTML."""
        m = folium.Map(location=[NY_ce_lat, NY_ce_lon], tiles="OpenStreetMap", zoom_start=11)
        m.choropleth(geo_str=str(fc),
                     data=data,
                     columns=("region", time),
                     key_on="feature.id",
                     threshold_scale=[5, 100, 300, 1000, 2000],
                     fill_color="OrRd",
                     fill_opacity=0.8,
                     line_opacity=0.3,
                     line_color="white",
                     reset=True)
        html = m._repr_html_()
        srcdoc = html.replace('"', '&quot;')
        embed = HTML('<iframe srcdoc="{}" '
                     'style="width: {}px; height: {}px; '
                     'border: none"></iframe>'.format(srcdoc, width, height))
        return embed

    def handle_submit_data(sender, data, tval):
        value = tval
        clear_output()

        start = data.columns[1]
        end   = data.columns[-1]
        time = None
        try:
            time = pd.Timestamp(value)
            assert start <= time <= end
            display(inline_map(data, time))
        except (ValueError, AssertionError):
            print("input should be date between {} and {} in format %Y-%m-%d %H:%M:%S".format(str(start), str(end)))
            return

    def display_data_map():
        tmap = widgets.Text("2016-06-15 15:00:00", description="time")
        bmap = widgets.Button(description="Show")
        bmap.on_click(lambda x: handle_submit_data(x, dfcT, tmap.value))
        display(tmap, bmap)

    def display_pred_map():
        tmap = widgets.Text("2016-06-15 15:00:00", description="time")
        bmap = widgets.Button(description="Show")
        bmap.on_click(lambda x: handle_submit_data(x, predT, tmap.value))
        display(tmap, bmap)

    def trace(typ, resample, region, frm, to):
        data = None
        desc = None
        if typ == "data":
            data = dfc
            desc = "data "
        elif typ == "pred":
            data = pred
            desc = "pred "
        else:
            data = preds[typ]
            desc = "pred "

        data = data.resample(resample[0]).sum().loc[frm:to, :]

        return go.Scatter(
            x=data.index, y=data[region],
            mode = 'lines',
            name = desc + str(region),
            line = dict(
                width = 1.5
                ),
            )

    def layout(typ, resample):
        buttons = [
            dict(count=1,
                 label='d',
                 step='day',
                 stepmode='todate'),
            dict(count=7,
                 label='w',
                 step='day',
                 stepmode='backwards'),
            dict(count=4,
                 label='m',
                 step='week',
                 stepmode='backwards'),
            dict(count=1,
                 label='y',
                 step='year',
                 stepmode='backwards'),
            dict(step="all")
        ]
        i = {"hour":0, "day":1, "week":2}
        n = {"data":"Historical", "pred":"Prediction", 1:"Prediction +1", 2:"Prediction +2",
             3:"Prediction +3", 4:"Prediction +4", 5:"Prediction +5", 6:"Prediction +6"}
        return go.Layout(
            title='%s %sly Data' % (n[typ], resample.title()),
            xaxis = dict(
                rangeslider=dict(),
                rangeselector=dict(
                    buttons=buttons[i[resample]:]
                    ),
                )
            )

    def plot_data(resample, regions, frm, to):
        traces = []
        for reg in regions:
            traces.append(trace("data", resample, reg, frm, to))

        lay = layout("data", resample)
        fig = go.Figure(data=traces, layout=lay)

        iplot(fig)

    def resample_select():
        return widgets.SelectionSlider(
            options=["hour", "day", "week"],
            value="hour",
            description="resample",
            desabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True)

    def region_select():
        return widgets.SelectionSlider(
            options=list(dfc.columns),
            value=1230,
            description='region',
            disabled=False
        )

    def regions_select():
        return widgets.SelectMultiple(
            options=list(dfc.columns),
            value=[1230, 1284],
            description='regions',
            disabled=False
        )

    def show_hist_check():
        return widgets.Checkbox(
            description='Show hist',
        )

    def hour_select():
        return widgets.SelectionSlider(
            options=list(range(1, 7)),
            value=4,
            description='hour',
            disabled=False
        )

    def show_hist_data():
        frmval = widgets.Text("2016-06-01 00:00:00", description="from")
        toval  = widgets.Text("2016-06-30 23:00:00", description="to")

        sval = resample_select()
        rvals = regions_select()
        bval = widgets.Button(description='Show')

        def handle_click_plot(sender):
            res     = sval.value
            frm     = frmval.value
            to      = toval.value
            regions = rvals.value
            clear_output()

            start = pred.index[0]
            end   = pred.index[-1]
            time  = None
            try:
                frm = pd.Timestamp(frm)
                to  = pd.Timestamp(to)
                assert start <= frm < to <= end
                plot_data(res, regions, frm, to)
            except (ValueError, AssertionError):
                print("inputs should be date between {} and {} in format %Y-%m-%d %H:%M:%S".format(str(start), str(end)))
                return

        bval.on_click(handle_click_plot)
        display(sval, frmval, toval, rvals, bval)

    def plot_pred(resample, region, frm, to, show_hist_check):
        traces = []
        traces.append(trace("pred", resample, region, frm, to))
        if show_hist_check:
            traces.append(trace("data", resample, region, frm, to))

        lay = layout("pred", resample)
        fig = go.Figure(data=traces, layout=lay)

        iplot(fig)

    def plot_predn(resample, region, hour, frm, to, show_hist_check):
        traces = []
        traces.append(trace(hour, resample, region, frm, to))
        if show_hist_check:
            traces.append(trace("data", resample, region, frm, to))

        lay = layout("pred", resample)
        fig = go.Figure(data=traces, layout=lay)

        iplot(fig)

    def show_pred_data():
        frmval = widgets.Text("2016-06-01 00:00:00", description="from")
        toval  = widgets.Text("2016-06-30 23:00:00", description="to")

        sval = resample_select()
        rval = region_select()
        bval = widgets.Button(description='Show')
        hchk = show_hist_check()
        def handle_click_plot(sender):
            res     = sval.value
            frm     = frmval.value
            to      = toval.value
            region  = rval.value
            hval    = hchk.value
            clear_output()

            start = pred.index[0]
            end   = pred.index[-1]
            time  = None
            try:
                frm = pd.Timestamp(frm)
                to  = pd.Timestamp(to)
                assert start <= frm < to <= end
                plot_pred(res, region, frm, to, hval)
            except (ValueError, AssertionError):
                print("inputs should be date between {} and {} in format %Y-%m-%d %H:%M:%S".format(str(start), str(end)))
                return

        bval.on_click(handle_click_plot)
        display(sval, frmval, toval, rval, hchk, bval)

    def show_predn_data():
        frmval = widgets.Text("2016-06-01 00:00:00", description="from")
        toval  = widgets.Text("2016-06-30 23:00:00", description="to")

        sval = resample_select()
        rval = region_select()
        bval = widgets.Button(description='Show')
        hsel = hour_select()
        hchk = show_hist_check()
        def handle_click_plot(sender):
            res     = sval.value
            frm     = frmval.value
            to      = toval.value
            region  = rval.value
            hval    = hchk.value
            hsvl    = hsel.value
            clear_output()

            start = pred.index[0]
            end   = pred.index[-1]
            time  = None
            try:
                frm = pd.Timestamp(frm)
                to  = pd.Timestamp(to)
                assert start <= frm < to <= end
                plot_predn(res, region, hsvl, frm, to, hval)
            except (ValueError, AssertionError):
                print("inputs should be date between {} and {} in format %Y-%m-%d %H:%M:%S".format(str(start), str(end)))
                return

        bval.on_click(handle_click_plot)
        display(sval, hsel, frmval, toval, rval, hchk, bval)

    def show_clusters():
        traces = []

        for i in range(N_clusters):
            traces.append(go.Scatter(
                x = df_2d_mds.reset_index()[clusters.reset_index()["cluster"] == i][0],
                y = df_2d_mds.reset_index()[clusters.reset_index()["cluster"] == i][1],
                text = dfc.columns,
                mode = 'markers',
                name = "cluster " + str(i),
                marker = dict(
                    size = 10
                )
            )
        )

        layout = go.Layout(
                title = ' MDS visualization (values) clusters',
                hovermode = 'closest'
            )

        data = traces
        fig = go.Figure(data=data, layout=layout)
        iplot(fig)

    def show_clusters_reps():
        show_points = 7*30

        start_date_show = "2016-04-24"
        end_date_show   = "2016-04-30"

        traces = []

        for i in range(N_clusters):
            traces.append(go.Scatter(
                x = df[start_date_show:end_date_show].index,
                y = df.loc[start_date_show:end_date_show, clusters_repr[i]],
                mode = 'lines',
                name = "cluster " + str(i),
                line = dict(
                    width = 1.5
                ),
                marker = dict(
                    size = 10
                )
            )
        )

        layout = go.Layout(
                title='Hourly normalized values of clusters centers',
            )

        data = traces
        fig = go.Figure(data=data, layout=layout)
        iplot(fig)

    def show_map_clusters():
        m = folium.Map(location=[NY_ce_lat, NY_ce_lon], tiles="OpenStreetMap", zoom_start=11)
        m.choropleth(geo_str=str(fc),
                 data=clusters.reset_index(),
                 columns=("region", "cluster"),
                 key_on="feature.id",
                 threshold_scale=list(range(5)),
                 fill_color="OrRd",
                 fill_opacity=0.8,
                 line_opacity=0.3,
                 line_color="white",
                 reset=True)
        html = m._repr_html_()
        srcdoc = html.replace('"', '&quot;')
        embed = HTML('<iframe srcdoc="{}" '
                     'style="width: {}px; height: {}px; '
                     'border: none"></iframe>'.format(srcdoc, 980, 600))
        return embed

    return (display_data_map,
     show_hist_data,
     feat,
     display_pred_map,
     show_pred_data,
     show_clusters,
     show_clusters_reps,
     show_map_clusters,
     show_predn_data)

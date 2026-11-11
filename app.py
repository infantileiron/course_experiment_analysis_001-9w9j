import argparse, csv, io, json, os
from pathlib import Path
from flask import Flask, jsonify, render_template, request, Response
from analysis import demo_events, validate_events, validate_config, analyze
from storage import Store, fingerprint

def create_app(data_dir=None):
    app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=32*1024*1024
    store=Store(data_dir or os.environ.get('APP_DATA_DIR',str(Path(__file__).parent/'data')))
    app.store=store
    if not store.list('datasets'):
        for scenario,title in [('null','无效果'),('uplift','真实提升'),('ratio_anomaly','分组异常')]:
            rows=demo_events(scenario)
            store.put('datasets',{'name':title,'rows':rows,'fingerprint':fingerprint(rows),'scenario':scenario},'demo_'+scenario)
    @app.errorhandler(ValueError)
    def invalid(exc):return jsonify(error=str(exc)),400
    @app.errorhandler(413)
    def too_large(exc):return jsonify(error='文件超过32MB限制'),413
    @app.get('/')
    def home():return render_template('index.html')
    @app.get('/api/datasets')
    def datasets():return jsonify([{k:v for k,v in d.items() if k!='rows'}|{'row_count':len(d['rows'])} for d in store.list('datasets')])
    @app.get('/api/datasets/<identifier>/download')
    def dataset_download(identifier):
        d=store.get('datasets',identifier);output=io.StringIO();writer=csv.DictWriter(output,fieldnames=['user_id','group','event','time','minutes','segment']);writer.writeheader();writer.writerows(d['rows'])
        return Response(output.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=experiment_events.csv'})
    @app.post('/api/datasets')
    def upload():
        f=request.files.get('file')
        if not f:raise ValueError('请选择CSV文件')
        try:rows=list(csv.DictReader(io.StringIO(f.read().decode('utf-8-sig'))))
        except UnicodeError:raise ValueError('CSV必须使用UTF-8编码')
        frame=validate_events(rows);frame['time']=frame.time.map(lambda x:x.isoformat());rows=frame.to_dict('records')
        d=store.put('datasets',{'name':request.form.get('name') or f.filename,'rows':rows,'fingerprint':fingerprint(rows)})
        return jsonify(id=d['id'],fingerprint=d['fingerprint']),201
    @app.get('/api/experiments')
    def experiments():return jsonify(store.list('experiments'))
    @app.get('/api/experiments/<identifier>')
    def experiment_get(identifier):return jsonify(store.get('experiments',identifier))
    @app.post('/api/experiments')
    def experiment_create():
        c=validate_config(request.get_json() or {});d=store.get('datasets',c['dataset_id'])
        c.update(dataset_fingerprint=d['fingerprint'],locked=True)
        return jsonify(store.put('experiments',c)),201
    @app.post('/api/analyze/<identifier>')
    def run(identifier):
        c=store.get('experiments',identifier);d=store.get('datasets',c['dataset_id'])
        explore=(request.get_json(silent=True) or {}).get('explore_segment')
        result=analyze(d['rows'],c,explore_segment=explore)
        return jsonify(store.put('runs',{'experiment_id':identifier,'experiment_name':c['name'],'dataset_id':d['id'],'dataset_fingerprint':d['fingerprint'],'config_snapshot':c,'result':result}))
    @app.get('/api/runs')
    def runs():return jsonify([{k:v for k,v in r.items() if k not in ['result','config_snapshot']} for r in store.list('runs')])
    @app.get('/api/runs/<identifier>')
    def get_run(identifier):return jsonify(store.get('runs',identifier))
    @app.get('/api/export/<identifier>')
    def export(identifier):
        r=store.get('runs',identifier)
        return Response(json.dumps(r,ensure_ascii=False,indent=2),mimetype='application/json',headers={'Content-Disposition':'attachment; filename=experiment_conclusion_'+identifier+'.json'})
    return app

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8101);args=parser.parse_args()
    create_app().run(host='127.0.0.1',port=args.port,debug=False)

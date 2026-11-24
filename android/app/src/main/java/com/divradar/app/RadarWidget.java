package com.divradar.app;

import android.app.PendingIntent;
import android.appwidget.AppWidgetManager;
import android.appwidget.AppWidgetProvider;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.view.View;
import android.widget.RemoteViews;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** 홈 화면 위젯: 준비 점수 상위 10종목 (매시간 자동 갱신 + 새로고침 버튼) */
public class RadarWidget extends AppWidgetProvider {

    static final String PREFS = "radar";
    static final String ACT_REFRESH = "com.divradar.app.REFRESH";

    private static final int[] ROW = {R.id.row1, R.id.row2, R.id.row3, R.id.row4, R.id.row5, R.id.row6, R.id.row7, R.id.row8, R.id.row9, R.id.row10};
    private static final int[] NAME = {R.id.name1, R.id.name2, R.id.name3, R.id.name4, R.id.name5, R.id.name6, R.id.name7, R.id.name8, R.id.name9, R.id.name10};
    private static final int[] TAG = {R.id.tag1, R.id.tag2, R.id.tag3, R.id.tag4, R.id.tag5, R.id.tag6, R.id.tag7, R.id.tag8, R.id.tag9, R.id.tag10};
    private static final int[] SCORE = {R.id.score1, R.id.score2, R.id.score3, R.id.score4, R.id.score5, R.id.score6, R.id.score7, R.id.score8, R.id.score9, R.id.score10};
    private static final int[] FIN = {R.id.fin1, R.id.fin2, R.id.fin3, R.id.fin4, R.id.fin5, R.id.fin6, R.id.fin7, R.id.fin8, R.id.fin9, R.id.fin10};

    @Override
    public void onUpdate(Context ctx, AppWidgetManager mgr, int[] ids) {
        fetchInReceiver(ctx);
    }

    @Override
    public void onReceive(Context ctx, Intent intent) {
        super.onReceive(ctx, intent);
        if (ACT_REFRESH.equals(intent.getAction())) fetchInReceiver(ctx);
    }

    private void fetchInReceiver(Context ctx) {
        final PendingResult pr = goAsync();
        final Context app = ctx.getApplicationContext();
        renderAll(app, prefs(app).getString("data", null)); // 일단 저장된 결과로 먼저 그림
        new Thread(() -> {
            try {
                fetchAndRender(app);
            } finally {
                pr.finish();
            }
        }).start();
    }

    static void refreshAsync(Context ctx) {
        final Context app = ctx.getApplicationContext();
        new Thread(() -> fetchAndRender(app)).start();
    }

    static SharedPreferences prefs(Context ctx) {
        return ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
    }

    static void fetchAndRender(Context ctx) {
        String repo = prefs(ctx).getString("repo", "");
        String json = null;
        if (!repo.isEmpty()) {
            json = get("https://api.github.com/repos/" + repo + "/contents/state/results.json", true);
            if (json == null) {
                json = get("https://raw.githubusercontent.com/" + repo + "/main/state/results.json?t="
                        + System.currentTimeMillis(), false);
            }
            if (json != null) prefs(ctx).edit().putString("data", json).apply();
        }
        renderAll(ctx, json != null ? json : prefs(ctx).getString("data", null));
    }

    private static String get(String url, boolean api) {
        HttpURLConnection c = null;
        try {
            c = (HttpURLConnection) new URL(url).openConnection();
            c.setConnectTimeout(10000);
            c.setReadTimeout(15000);
            c.setRequestProperty("User-Agent", "DivRadar-Android");
            if (api) c.setRequestProperty("Accept", "application/vnd.github.raw");
            c.setUseCaches(false);
            if (c.getResponseCode() != 200) return null;
            InputStream in = c.getInputStream();
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
            String s = out.toString(StandardCharsets.UTF_8.name());
            new JSONObject(s); // 형식 확인
            return s;
        } catch (Exception e) {
            return null;
        } finally {
            if (c != null) c.disconnect();
        }
    }

    static void renderAll(Context ctx, String json) {
        AppWidgetManager mgr = AppWidgetManager.getInstance(ctx);
        int[] ids = mgr.getAppWidgetIds(new ComponentName(ctx, RadarWidget.class));
        if (ids == null || ids.length == 0) return;
        RemoteViews rv = build(ctx, json);
        for (int id : ids) mgr.updateAppWidget(id, rv);
    }

    private static PendingIntent openApp(Context ctx, String code, int req) {
        Intent i = new Intent(ctx, MainActivity.class);
        i.setAction(Intent.ACTION_VIEW);
        i.setData(Uri.parse(code == null ? "divscan://open" : "divscan://open?code=" + code));
        i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        return PendingIntent.getActivity(ctx, req, i,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
    }

    private static RemoteViews build(Context ctx, String json) {
        RemoteViews rv = new RemoteViews(ctx.getPackageName(), R.layout.widget);
        rv.setOnClickPendingIntent(R.id.root, openApp(ctx, null, 100));

        Intent r = new Intent(ctx, RadarWidget.class);
        r.setAction(ACT_REFRESH);
        rv.setOnClickPendingIntent(R.id.refresh, PendingIntent.getBroadcast(ctx, 101, r,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE));

        List<JSONObject> items = new ArrayList<>();
        String time = "";
        try {
            if (json != null) {
                JSONObject d = new JSONObject(json);
                String up = d.optString("updated", "");
                if (up.length() >= 16) time = up.substring(5, 10).replace('-', '/') + " " + up.substring(11, 16);
                JSONArray arr = d.optJSONArray("items");
                if (arr != null) for (int i = 0; i < arr.length(); i++) items.add(arr.getJSONObject(i));
                Collections.sort(items, (a, b) -> b.optInt("ready") - a.optInt("ready"));
            }
        } catch (Exception ignored) {
        }
        rv.setTextViewText(R.id.time, time);

        if (items.isEmpty()) {
            rv.setViewVisibility(R.id.empty, View.VISIBLE);
            rv.setTextViewText(R.id.empty, prefs(ctx).getString("repo", "").isEmpty()
                    ? "앱 설정에서 GitHub 저장소를 연결해 주세요" : "아직 결과가 없어요");
        } else {
            rv.setViewVisibility(R.id.empty, View.GONE);
        }

        for (int i = 0; i < ROW.length; i++) {
            if (i >= items.size()) {
                rv.setViewVisibility(ROW[i], View.GONE);
                continue;
            }
            JSONObject it = items.get(i);
            rv.setViewVisibility(ROW[i], View.VISIBLE);
            rv.setTextViewText(NAME[i], it.optString("name"));
            int sig = it.optInt("signal", 0);
            String type = it.optString("type", "");
            rv.setTextViewText(TAG[i], sig <= 0 ? "" : ("대기".equals(type) ? "대기" : "신호") + sig);
            rv.setTextColor(TAG[i], "대기".equals(type) ? 0xFF9AA8FF : 0xFFFFCF5A);
            JSONObject fin = it.optJSONObject("fin");
            String g = fin == null ? "" : fin.optString("grade", "");
            if ("위험".equals(g)) {
                rv.setTextViewText(FIN[i], "⚠위험");
                rv.setTextColor(FIN[i], 0xFFFF5C62);
            } else if (g.length() == 1) {
                rv.setTextViewText(FIN[i], "재무" + g);
                rv.setTextColor(FIN[i], "A".equals(g) ? 0xFF36D399 : "B".equals(g) ? 0xFF7AA9FF : "C".equals(g) ? 0xFFF5C451 : 0xFFFF9A5A);
            } else {
                rv.setTextViewText(FIN[i], "");
            }
            int ready = it.optInt("ready", 0);
            rv.setTextViewText(SCORE[i], String.valueOf(ready));
            rv.setTextColor(SCORE[i], ready >= 60 ? 0xFF36D399 : ready >= 40 ? 0xFFF5C451 : 0xFF8A96A6);
            rv.setOnClickPendingIntent(ROW[i], openApp(ctx, it.optString("code"), 110 + i));
        }
        return rv;
    }
}

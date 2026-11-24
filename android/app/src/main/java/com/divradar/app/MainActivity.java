package com.divradar.app;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Bundle;
import android.webkit.JavascriptInterface;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

public class MainActivity extends Activity {

    private WebView web;
    private boolean loaded = false;
    private String pendingCode = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(0xFF0C0F14);
        getWindow().setNavigationBarColor(0xFF0C0F14);

        web = new WebView(this);
        web.setBackgroundColor(0xFF0C0F14);
        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);
        s.setAllowFileAccess(true);
        s.setAllowUniversalAccessFromFileURLs(true);
        s.setCacheMode(WebSettings.LOAD_NO_CACHE);
        web.addJavascriptInterface(new Bridge(this), "Android");
        web.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                loaded = true;
                flushPending();
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest req) {
                Uri u = req.getUrl();
                if ("file".equals(u.getScheme())) return false;
                openExternal(MainActivity.this, u.toString());
                return true;
            }
        });
        setContentView(web);
        handleIntent(getIntent());
        web.loadUrl("file:///android_asset/index.html");
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        handleIntent(intent);
        flushPending();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (loaded) web.evaluateJavascript("window.reload && window.reload()", null);
    }

    private void handleIntent(Intent intent) {
        if (intent == null || intent.getData() == null) return;
        String code = intent.getData().getQueryParameter("code");
        if (code != null && code.matches("\\d{6}")) pendingCode = code;
    }

    private void flushPending() {
        if (loaded && pendingCode != null) {
            web.evaluateJavascript("window.openCode && window.openCode('" + pendingCode + "')", null);
            pendingCode = null;
        }
    }

    @Override
    @SuppressWarnings("deprecation")
    public void onBackPressed() {
        if (!loaded) {
            super.onBackPressed();
            return;
        }
        web.evaluateJavascript("window.onBack ? window.onBack() : false", value -> {
            if (!"true".equals(value)) MainActivity.super.onBackPressed();
        });
    }

    static void openExternal(Context ctx, String url) {
        try {
            Intent i = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
            i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            ctx.startActivity(i);
        } catch (Exception ignored) {
        }
    }

    /** 화면(자바스크립트)에서 부르는 기능 */
    static class Bridge {
        private final Context ctx;

        Bridge(Context ctx) {
            this.ctx = ctx.getApplicationContext();
        }

        private SharedPreferences prefs() {
            return ctx.getSharedPreferences(RadarWidget.PREFS, Context.MODE_PRIVATE);
        }

        @JavascriptInterface
        public String getRepo() {
            return prefs().getString("repo", "");
        }

        @JavascriptInterface
        public void saveRepo(String repo) {
            prefs().edit().putString("repo", repo == null ? "" : repo.trim()).apply();
            RadarWidget.refreshAsync(ctx);
        }

        @JavascriptInterface
        public void saveData(String json) {
            prefs().edit().putString("data", json).apply();
            RadarWidget.renderAll(ctx, json);
        }

        @JavascriptInterface
        public void openUrl(String url) {
            if (url != null && url.startsWith("https://")) openExternal(ctx, url);
        }
    }
}

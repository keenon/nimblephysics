var gulp = require("gulp");
var sass = require("gulp-sass");
var concat = require("gulp-concat");
var maps = require("gulp-sourcemaps");
var uglify = require("gulp-uglify");
var rename = require("gulp-rename");
var server = require("browser-sync").create();

function reload(done) {
  server.reload();
  done();
}

function serve(done) {
  server.init({
    server: {
      baseDir: "app",
    },
  });
  done();
}

// compile sass to css
function css() {
  return gulp
    .src("app/assets/scss/style.scss")
    .pipe(maps.init())
    .pipe(sass())
    .pipe(maps.write("./"))
    .pipe(gulp.dest("app/assets/css"))
    .pipe(server.stream());
}

// concatenate js files
function scripts() {
  return gulp
    .src([
      "node_modules/jquery/dist/jquery.js",
      "node_modules/bootstrap/dist/js/bootstrap.bundle.js",
      "node_modules/owl.carousel/dist/owl.carousel.js",
      "node_modules/magnific-popup/dist/jquery.magnific-popup.min.js",
      "node_modules/swiper/dist/js/swiper.js",
      "node_modules/masonry-layout/dist/masonry.pkgd.js",
      "node_modules/sticky-kit/dist/sticky-kit.js",
      "node_modules/headroom.js/dist/headroom.js",
      "node_modules/headroom.js/dist/jQuery.headroom.js",
      "node_modules/skrollr/dist/skrollr.min.js",
      "node_modules/smooth-scroll/dist/smooth-scroll.js",
      "node_modules/lavalamp/js/jquery.lavalamp.min.js",
      "node_modules/bootstrap-select/dist/js/bootstrap-select.min.js",
      "node_modules/clipboard/dist/clipboard.min.js",
      "node_modules/prismjs/prism.js",
      "node_modules/prismjs/components/prism-python.js",
      "node_modules/prismjs/plugins/toolbar/prism-toolbar.js",
      "node_modules/prismjs/plugins/copy-to-clipboard/prism-copy-to-clipboard.min.js",
      "node_modules/video.js/dist/video.js",
      "node_modules/videojs-youtube/dist/Youtube.js",
      "app/assets/js/modernizr.js",
    ])
    .pipe(maps.init())
    .pipe(concat("vendor.js"))
    .pipe(maps.write("./"))
    .pipe(gulp.dest("app/assets/js"));
}

// concatenate css files
function styles() {
  return gulp
    .src([
      "node_modules/swiper/dist/css/swiper.css",
      "node_modules/owl.carousel/dist/assets/owl.carousel.css",
      "node_modules/magnific-popup/dist/magnific-popup.css",
      "node_modules/bootstrap-select/dist/css/bootstrap-select.css",
      "node_modules/prismjs/themes/prism.css",
      "node_modules/prismjs/plugins/toolbar/prism-toolbar.css",
      "node_modules/prismjs/plugins/copy-to-clipboard/prism-copy-to-clipboard.min.js",
      "node_modules/video.js/dist/video-js.css",
    ])
    .pipe(maps.init({ loadMaps: true }))
    .pipe(concat("vendor.css"))
    .pipe(maps.write())
    .pipe(gulp.dest("app/assets/css"));
}

// minify js
function minify() {
  return gulp
    .src("app/assets/js/vendor.js")
    .pipe(maps.init())
    .pipe(uglify())
    .pipe(rename("vendor.min.js"))
    .pipe(maps.write("./"))
    .pipe(gulp.dest("app/assets/js"));
}

// watch for changes
function watch() {
  gulp.watch("app/assets/scss/**/*", css);
  gulp.watch(["app/*", "app/**/*", "app/assets/js/*"], reload);
  gulp.watch("gulpfile.js", gulp.series(scripts, styles, minify));
}

const build = gulp.series(css, scripts, styles, gulp.parallel(watch, serve));

// tasks
exports.css = css;
exports.scripts = scripts;
exports.styles = styles;
exports.minify = minify;

exports.default = build;

$(document).ready(function(){
	var fform=$("form");
	$.ajax({
                url: "information",
                dataType: "json",
                type:"POST",
                success: function (fform) {
                	if(fform.val()=='')
                	{
                		$("#submit").attr("value","SUBMIT");
                		console.log($("#male").attr("checked"));
						fform=$("#fform");
						$("#submit").submit(function(){
							$.ajax({
				                url: "information",
				                data:JSON.stringify(fform),
				                dataType: "json",
				                type:"POST",
				                success: function (result) {
				                }
				            });
						});
                	}
                	else
                	{
	        			for(var i=0;i<fform.length;++i)
	                	{
	                		var s=fform[i].name;
	                		var cnt=fform[i].val();
	                		$("input[name=s]").attr("value",cnt);
	                	}
                		$("input").attr("disabled",true);
                		$("#submit").attr("value","EDIT");
                		$("#submit").click(function(){
                			$("input").attr("disabled",false);
                			$("#submit").attr("value","SAVE");
                				fform=$("#fform");
                				$("#submit").submit(function(){
								$.ajax({
				                url: "information",
				                data:JSON.stringify(fform),
				                dataType: "json",
				                type:"POST",
				                success: function (result) {
				                }
				            });
						});

                		});
                	}
                }
            });
	$("input").focus(function(){
		$(this).parent().css("border","2px solid #179b77");
	});
	$("input").blur(function(){
		$(this).parent().css("border","");
	});
});